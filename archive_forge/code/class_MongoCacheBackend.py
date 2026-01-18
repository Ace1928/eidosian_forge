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
class MongoCacheBackend(api.CacheBackend):
    """A MongoDB based caching backend implementing dogpile backend APIs.

    Arguments accepted in the arguments dictionary:

    :param db_hosts: string (required), hostname or IP address of the
        MongoDB server instance. This can be a single MongoDB connection URI,
        or a list of MongoDB connection URIs.

    :param db_name: string (required), the name of the database to be used.

    :param cache_collection: string (required), the name of collection to store
        cached data.
        *Note:* Different collection name can be provided if there is need to
        create separate container (i.e. collection) for cache data. So region
        configuration is done per collection.

    Following are optional parameters for MongoDB backend configuration,

    :param username: string, the name of the user to authenticate.

    :param password: string, the password of the user to authenticate.

    :param max_pool_size: integer, the maximum number of connections that the
        pool will open simultaneously. By default the pool size is 10.

    :param w: integer, write acknowledgement for MongoDB client

        If not provided, then no default is set on MongoDB and then write
        acknowledgement behavior occurs as per MongoDB default. This parameter
        name is same as what is used in MongoDB docs. This value is specified
        at collection level so its applicable to `cache_collection` db write
        operations.

        If this is a replica set, write operations will block until they have
        been replicated to the specified number or tagged set  of servers.
        Setting w=0 disables write acknowledgement and all other write concern
        options.

    :param read_preference: string, the read preference mode for MongoDB client
        Expected value is ``primary``, ``primaryPreferred``, ``secondary``,
        ``secondaryPreferred``, or ``nearest``. This read_preference is
        specified at collection level so its applicable to `cache_collection`
        db read operations.

    :param use_replica: boolean, flag to indicate if replica client to be
        used. Default is `False`. `replicaset_name` value is required if
        `True`.

    :param replicaset_name: string, name of replica set.
        Becomes required if `use_replica` is `True`

    :param son_manipulator: string, name of class with module name which
        implements MongoDB SONManipulator.
        Default manipulator used is :class:`.BaseTransform`.

        This manipulator is added per database. In multiple cache
        configurations, the manipulator name should be same if same
        database name ``db_name`` is used in those configurations.

        SONManipulator is used to manipulate custom data types as they are
        saved or retrieved from MongoDB. Custom impl is only needed if cached
        data is custom class and needs transformations when saving or reading
        from db. If dogpile cached value contains built-in data types, then
        BaseTransform class is sufficient as it already handles dogpile
        CachedValue class transformation.

    :param mongo_ttl_seconds: integer, interval in seconds to indicate maximum
        time-to-live value.
        If value is greater than 0, then its assumed that cache_collection
        needs to be TTL type (has index at 'doc_date' field).
        By default, the value is -1 and its disabled.
        Reference: <http://docs.mongodb.org/manual/tutorial/expire-data/>

        .. NOTE::

            This parameter is different from Dogpile own
            expiration_time, which is the number of seconds after which Dogpile
            will consider the value to be expired. When Dogpile considers a
            value to be expired, it continues to use the value until generation
            of a new value is complete, when using CacheRegion.get_or_create().
            Therefore, if you are setting `mongo_ttl_seconds`, you will want to
            make sure it is greater than expiration_time by at least enough
            seconds for new values to be generated, else the value would not
            be available during a regeneration, forcing all threads to wait for
            a regeneration each time a value expires.

    :param ssl: boolean, If True, create the connection to the server
        using SSL. Default is `False`. Client SSL connection parameters depends
        on server side SSL setup. For further reference on SSL configuration:
        <http://docs.mongodb.org/manual/tutorial/configure-ssl/>

    :param ssl_keyfile: string, the private keyfile used to identify the
        local connection against mongod. If included with the certfile then
        only the `ssl_certfile` is needed. Used only when `ssl` is `True`.

    :param ssl_certfile: string, the certificate file used to identify the
        local connection against mongod. Used only when `ssl` is `True`.

    :param ssl_ca_certs: string, the ca_certs file contains a set of
        concatenated 'certification authority' certificates, which are used to
        validate certificates passed from the other end of the connection.
        Used only when `ssl` is `True`.

    :param ssl_cert_reqs: string, the parameter cert_reqs specifies whether
        a certificate is required from the other side of the connection, and
        whether it will be validated if provided. It must be one of the three
        values ``ssl.CERT_NONE`` (certificates ignored), ``ssl.CERT_OPTIONAL``
        (not required, but validated if provided), or
        ``ssl.CERT_REQUIRED`` (required and validated). If the value of this
        parameter is not ``ssl.CERT_NONE``, then the ssl_ca_certs parameter
        must point to a file of CA certificates. Used only when `ssl`
        is `True`.

    Rest of arguments are passed to mongo calls for read, write and remove.
    So related options can be specified to pass to these operations.

    Further details of various supported arguments can be referred from
    <http://api.mongodb.org/python/current/api/pymongo/>

    """

    def __init__(self, arguments):
        self.api = MongoApi(arguments)

    @dp_util.memoized_property
    def client(self):
        """Initializes MongoDB connection and collection defaults.

        This initialization is done only once and performed as part of lazy
        inclusion of MongoDB dependency i.e. add imports only if related
        backend is used.

        :return: :class:`.MongoApi` instance
        """
        self.api.get_cache_collection()
        return self.api

    def get(self, key):
        """Retrieves the value for a key.

        :param key: key to be retrieved.
        :returns: value for a key or :data:`oslo_cache.core.NO_VALUE`
            for nonexistent or expired keys.
        """
        value = self.client.get(key)
        if value is None:
            return _NO_VALUE
        else:
            return value

    def get_multi(self, keys):
        """Return multiple values from the cache, based on the given keys.

        :param keys: sequence of keys to be retrieved.
        :returns: returns values (or :data:`oslo_cache.core.NO_VALUE`)
            as a list matching the keys given.
        """
        values = self.client.get_multi(keys)
        return [_NO_VALUE if key not in values else values[key] for key in keys]

    def set(self, key, value):
        self.client.set(key, value)

    def set_multi(self, mapping):
        self.client.set_multi(mapping)

    def delete(self, key):
        self.client.delete(key)

    def delete_multi(self, keys):
        self.client.delete_multi(keys)