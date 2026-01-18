import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
class MongoApi(object):
    """Class handling MongoDB specific functionality.

    This class uses PyMongo APIs internally to create database connection
    with configured pool size, ensures unique index on key, does database
    authentication and ensure TTL collection index if configured so.
    This class also serves as handle to cache collection for dogpile cache
    APIs.

    In a single deployment, multiple cache configuration can be defined. In
    that case of multiple cache collections usage, db client connection pool
    is shared when cache collections are within same database.
    """
    _DB = {}
    _MONGO_COLLS = {}

    def __init__(self, arguments):
        self._init_args(arguments)
        self._data_manipulator = None

    def _init_args(self, arguments):
        """Helper logic for collecting and parsing MongoDB specific arguments.

        The arguments passed in are separated out in connection specific
        setting and rest of arguments are passed to create/update/delete
        db operations.
        """
        self.conn_kwargs = {}
        self.hosts = arguments.pop('db_hosts', None)
        if self.hosts is None:
            msg = _('db_hosts value is required')
            raise exception.ConfigurationError(msg)
        self.db_name = arguments.pop('db_name', None)
        if self.db_name is None:
            msg = _('database db_name is required')
            raise exception.ConfigurationError(msg)
        self.cache_collection = arguments.pop('cache_collection', None)
        if self.cache_collection is None:
            msg = _('cache_collection name is required')
            raise exception.ConfigurationError(msg)
        self.username = arguments.pop('username', None)
        self.password = arguments.pop('password', None)
        self.max_pool_size = arguments.pop('max_pool_size', 10)
        self.w = arguments.pop('w', -1)
        try:
            self.w = int(self.w)
        except ValueError:
            msg = _('integer value expected for w (write concern attribute)')
            raise exception.ConfigurationError(msg)
        self.read_preference = arguments.pop('read_preference', None)
        self.use_replica = arguments.pop('use_replica', False)
        if self.use_replica:
            if arguments.get('replicaset_name') is None:
                msg = _('replicaset_name required when use_replica is True')
                raise exception.ConfigurationError(msg)
            self.replicaset_name = arguments.get('replicaset_name')
        self.son_manipulator = arguments.pop('son_manipulator', None)
        self.ttl_seconds = arguments.pop('mongo_ttl_seconds', -1)
        try:
            self.ttl_seconds = int(self.ttl_seconds)
        except ValueError:
            msg = _('integer value expected for mongo_ttl_seconds')
            raise exception.ConfigurationError(msg)
        self.conn_kwargs['ssl'] = arguments.pop('ssl', False)
        if self.conn_kwargs['ssl']:
            ssl_keyfile = arguments.pop('ssl_keyfile', None)
            ssl_certfile = arguments.pop('ssl_certfile', None)
            ssl_ca_certs = arguments.pop('ssl_ca_certs', None)
            ssl_cert_reqs = arguments.pop('ssl_cert_reqs', None)
            if ssl_keyfile:
                self.conn_kwargs['ssl_keyfile'] = ssl_keyfile
            if ssl_certfile:
                self.conn_kwargs['ssl_certfile'] = ssl_certfile
            if ssl_ca_certs:
                self.conn_kwargs['ssl_ca_certs'] = ssl_ca_certs
            if ssl_cert_reqs:
                self.conn_kwargs['ssl_cert_reqs'] = self._ssl_cert_req_type(ssl_cert_reqs)
        self.meth_kwargs = arguments

    def _ssl_cert_req_type(self, req_type):
        try:
            import ssl
        except ImportError:
            raise exception.ConfigurationError(_('no ssl support available'))
        req_type = req_type.upper()
        try:
            return {'NONE': ssl.CERT_NONE, 'OPTIONAL': ssl.CERT_OPTIONAL, 'REQUIRED': ssl.CERT_REQUIRED}[req_type]
        except KeyError:
            msg = _('Invalid ssl_cert_reqs value of %s, must be one of "NONE", "OPTIONAL", "REQUIRED"') % req_type
            raise exception.ConfigurationError(msg)

    def _get_db(self):
        global pymongo
        import pymongo
        if self.use_replica:
            connection = pymongo.MongoReplicaSetClient(host=self.hosts, replicaSet=self.replicaset_name, max_pool_size=self.max_pool_size, **self.conn_kwargs)
        else:
            connection = pymongo.MongoClient(host=self.hosts, max_pool_size=self.max_pool_size, **self.conn_kwargs)
        database = getattr(connection, self.db_name)
        self._assign_data_mainpulator()
        database.add_son_manipulator(self._data_manipulator)
        if self.username and self.password:
            database.authenticate(self.username, self.password)
        return database

    def _assign_data_mainpulator(self):
        if self._data_manipulator is None:
            if self.son_manipulator:
                self._data_manipulator = importutils.import_object(self.son_manipulator)
            else:
                self._data_manipulator = BaseTransform()

    def _get_doc_date(self):
        if self.ttl_seconds > 0:
            expire_delta = datetime.timedelta(seconds=self.ttl_seconds)
            doc_date = timeutils.utcnow() + expire_delta
        else:
            doc_date = timeutils.utcnow()
        return doc_date

    def get_cache_collection(self):
        if self.cache_collection not in self._MONGO_COLLS:
            global pymongo
            import pymongo
            if self.db_name not in self._DB:
                self._DB[self.db_name] = self._get_db()
            coll = getattr(self._DB[self.db_name], self.cache_collection)
            self._assign_data_mainpulator()
            if self.read_preference:
                f = getattr(pymongo.read_preferences, 'read_pref_mode_from_name', None)
                if not f:
                    f = pymongo.read_preferences.mongos_enum
                self.read_preference = f(self.read_preference)
                coll.read_preference = self.read_preference
            if self.w > -1:
                coll.write_concern['w'] = self.w
            if self.ttl_seconds > 0:
                kwargs = {'expireAfterSeconds': self.ttl_seconds}
                coll.ensure_index('doc_date', cache_for=5, **kwargs)
            else:
                self._validate_ttl_index(coll, self.cache_collection, self.ttl_seconds)
            self._MONGO_COLLS[self.cache_collection] = coll
        return self._MONGO_COLLS[self.cache_collection]

    def _get_cache_entry(self, key, value, meta, doc_date):
        """MongoDB cache data representation.

        Storing cache key as ``_id`` field as MongoDB by default creates
        unique index on this field. So no need to create separate field and
        index for storing cache key. Cache data has additional ``doc_date``
        field for MongoDB TTL collection support.
        """
        return dict(_id=key, value=value, meta=meta, doc_date=doc_date)

    def _validate_ttl_index(self, collection, coll_name, ttl_seconds):
        """Checks if existing TTL index is removed on a collection.

        This logs warning when existing collection has TTL index defined and
        new cache configuration tries to disable index with
        ``mongo_ttl_seconds < 0``. In that case, existing index needs
        to be addressed first to make new configuration effective.
        Refer to MongoDB documentation around TTL index for further details.
        """
        indexes = collection.index_information()
        for indx_name, index_data in indexes.items():
            if all((k in index_data for k in ('key', 'expireAfterSeconds'))):
                existing_value = index_data['expireAfterSeconds']
                fld_present = 'doc_date' in index_data['key'][0]
                if fld_present and existing_value > -1 and (ttl_seconds < 1):
                    msg = 'TTL index already exists on db collection <%(c_name)s>, remove index <%(indx_name)s> first to make updated mongo_ttl_seconds value to be  effective'
                    LOG.warning(msg, {'c_name': coll_name, 'indx_name': indx_name})

    def get(self, key):
        criteria = {'_id': key}
        result = self.get_cache_collection().find_one(spec_or_id=criteria, **self.meth_kwargs)
        if result:
            return result['value']
        else:
            return None

    def get_multi(self, keys):
        db_results = self._get_results_as_dict(keys)
        return {doc['_id']: doc['value'] for doc in db_results.values()}

    def _get_results_as_dict(self, keys):
        criteria = {'_id': {'$in': keys}}
        db_results = self.get_cache_collection().find(spec=criteria, **self.meth_kwargs)
        return {doc['_id']: doc for doc in db_results}

    def set(self, key, value):
        doc_date = self._get_doc_date()
        ref = self._get_cache_entry(key, value.payload, value.metadata, doc_date)
        spec = {'_id': key}
        ref = self._data_manipulator.transform_incoming(ref, self)
        self.get_cache_collection().find_and_modify(spec, ref, upsert=True, **self.meth_kwargs)

    def set_multi(self, mapping):
        """Insert multiple documents specified as key, value pairs.

        In this case, multiple documents can be added via insert provided they
        do not exist.
        Update of multiple existing documents is done one by one
        """
        doc_date = self._get_doc_date()
        insert_refs = []
        update_refs = []
        existing_docs = self._get_results_as_dict(list(mapping.keys()))
        for key, value in mapping.items():
            ref = self._get_cache_entry(key, value.payload, value.metadata, doc_date)
            if key in existing_docs:
                ref['_id'] = existing_docs[key]['_id']
                update_refs.append(ref)
            else:
                insert_refs.append(ref)
        if insert_refs:
            self.get_cache_collection().insert(insert_refs, manipulate=True, **self.meth_kwargs)
        for upd_doc in update_refs:
            self.get_cache_collection().save(upd_doc, manipulate=True, **self.meth_kwargs)

    def delete(self, key):
        criteria = {'_id': key}
        self.get_cache_collection().remove(spec_or_id=criteria, **self.meth_kwargs)

    def delete_multi(self, keys):
        criteria = {'_id': {'$in': keys}}
        self.get_cache_collection().remove(spec_or_id=criteria, **self.meth_kwargs)