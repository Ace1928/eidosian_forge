import datetime
from hashlib import sha1
import logging
from pymongo import MongoClient
import pymongo.errors
import pymongo.uri_parser
from saml2.eptid import Eptid
from saml2.ident import IdentDB
from saml2.ident import Unknown
from saml2.ident import code_binary
from saml2.mdie import from_dict
from saml2.mdie import to_dict
from saml2.mdstore import InMemoryMetaData
from saml2.mdstore import load_metadata_modules
from saml2.mdstore import metadata_modules
from saml2.s_utils import PolicyError
from saml2.saml import NAMEID_FORMAT_PERSISTENT
class MDB:
    primary_key = 'mdb'

    def __init__(self, database, collection, **kwargs):
        _db = _mdb_get_database(database, **kwargs)
        self.db = _db[collection]

    def store(self, value, **kwargs):
        if value:
            doc = {self.primary_key: value}
        else:
            doc = {}
        doc.update(kwargs)
        if 'created_at' not in doc:
            doc['created_at'] = datetime.datetime.utcnow()
        _ = self.db.insert_one(doc)

    def get(self, value=None, **kwargs):
        if value is not None:
            doc = {self.primary_key: value}
            doc.update(kwargs)
            return [item for item in self.db.find(doc)]
        elif kwargs:
            return [item for item in self.db.find(kwargs)]

    def remove(self, key=None, **kwargs):
        if key is None:
            if kwargs:
                self.db.delete_many(filter=kwargs)
        else:
            doc = {self.primary_key: key}
            doc.update(kwargs)
            self.db.delete_many(filter=doc)

    def keys(self):
        for item in self.db.find():
            yield item[self.primary_key]

    def items(self):
        for item in self.db.find():
            _key = item[self.primary_key]
            del item[self.primary_key]
            del item['_id']
            yield (_key, item)

    def __contains__(self, key):
        doc = {self.primary_key: key}
        res = [item for item in self.db.find(doc)]
        if not res:
            return False
        else:
            return True

    def reset(self):
        self.db.drop()