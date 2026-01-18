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
class EptidMDB(Eptid):

    def __init__(self, secret, database='', collection='eptid'):
        Eptid.__init__(self, secret)
        self.mdb = MDB(database, collection)
        self.mdb.primary_key = 'eptid_key'

    def __getitem__(self, key):
        res = self.mdb.get(key)
        if not res:
            raise KeyError(key)
        elif len(res) == 1:
            return res[0]['eptid']
        else:
            raise CorruptDatabase('Found more than one EPTID document')

    def __setitem__(self, key, value):
        self.mdb.store(key, **{'eptid': value})