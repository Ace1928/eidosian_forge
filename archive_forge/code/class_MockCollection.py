import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
class MockCollection(object):

    def __init__(self, db, name):
        super(MockCollection, self).__init__()
        self.name = name
        self._collection_database = db
        self._documents = {}
        self.write_concern = {}

    def __getattr__(self, name):
        if name == 'database':
            return self._collection_database

    def ensure_index(self, key_or_list, *args, **kwargs):
        pass

    def index_information(self):
        return {}

    def find_one(self, spec_or_id=None, *args, **kwargs):
        if spec_or_id is None:
            spec_or_id = {}
        if not isinstance(spec_or_id, collections.abc.Mapping):
            spec_or_id = {'_id': spec_or_id}
        try:
            return next(self.find(spec_or_id, *args, **kwargs))
        except StopIteration:
            return None

    def find(self, spec=None, *args, **kwargs):
        return MockCursor(self, functools.partial(self._get_dataset, spec))

    def _get_dataset(self, spec):
        dataset = (self._copy_doc(document, dict) for document in self._iter_documents(spec))
        return dataset

    def _iter_documents(self, spec=None):
        return (SON_MANIPULATOR.transform_outgoing(document, self) for document in self._documents.values() if self._apply_filter(document, spec))

    def _apply_filter(self, document, query):
        for key, search in query.items():
            doc_val = document.get(key)
            if isinstance(search, dict):
                op_dict = {'$in': lambda dv, sv: dv in sv}
                is_match = all((op_str in op_dict and op_dict[op_str](doc_val, search_val) for op_str, search_val in search.items()))
            else:
                is_match = doc_val == search
        return is_match

    def _copy_doc(self, obj, container):
        if isinstance(obj, list):
            new = []
            for item in obj:
                new.append(self._copy_doc(item, container))
            return new
        if isinstance(obj, dict):
            new = container()
            for key, value in list(obj.items()):
                new[key] = self._copy_doc(value, container)
            return new
        else:
            return copy.copy(obj)

    def insert(self, data, manipulate=True, **kwargs):
        if isinstance(data, list):
            return [self._insert(element) for element in data]
        return self._insert(data)

    def save(self, data, manipulate=True, **kwargs):
        return self._insert(data)

    def _insert(self, data):
        if '_id' not in data:
            data['_id'] = uuidutils.generate_uuid(dashed=False)
        object_id = data['_id']
        self._documents[object_id] = self._internalize_dict(data)
        return object_id

    def find_and_modify(self, spec, document, upsert=False, **kwargs):
        self.update(spec, document, upsert, **kwargs)

    def update(self, spec, document, upsert=False, **kwargs):
        existing_docs = [doc for doc in self._documents.values() if self._apply_filter(doc, spec)]
        if existing_docs:
            existing_doc = existing_docs[0]
            _id = existing_doc['_id']
            existing_doc.clear()
            existing_doc['_id'] = _id
            existing_doc.update(self._internalize_dict(document))
        elif upsert:
            existing_doc = self._documents[self._insert(document)]

    def _internalize_dict(self, d):
        return {k: copy.deepcopy(v) for k, v in d.items()}

    def remove(self, spec_or_id=None, search_filter=None):
        """Remove objects matching spec_or_id from the collection."""
        if spec_or_id is None:
            spec_or_id = search_filter if search_filter else {}
        if not isinstance(spec_or_id, dict):
            spec_or_id = {'_id': spec_or_id}
        to_delete = list(self.find(spec=spec_or_id))
        for doc in to_delete:
            doc_id = doc['_id']
            del self._documents[doc_id]
        return {'connectionId': uuidutils.generate_uuid(dashed=False), 'n': len(to_delete), 'ok': 1.0, 'err': None}