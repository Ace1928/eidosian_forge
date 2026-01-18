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
class IdentMDB(IdentDB):

    def __init__(self, database='', collection='ident', domain='', name_qualifier=''):
        IdentDB.__init__(self, None, domain, name_qualifier)
        self.mdb = MDB(database=database, collection=collection)
        self.mdb.primary_key = 'user_id'

    def in_store(self, _id):
        if [x for x in self.mdb.get(ident_id=_id)]:
            return True
        else:
            return False

    def create_id(self, nformat, name_qualifier='', sp_name_qualifier=''):
        _id = self._create_id(nformat, name_qualifier, sp_name_qualifier)
        while self.in_store(_id):
            _id = self._create_id(nformat, name_qualifier, sp_name_qualifier)
        return _id

    def store(self, ident, name_id):
        self.mdb.store(ident, name_id=to_dict(name_id, MMODS, True))

    def find_nameid(self, userid, nformat=None, sp_name_qualifier=None, name_qualifier=None, sp_provided_id=None, **kwargs):
        kwargs = {}
        if nformat:
            kwargs['name_format'] = nformat
        if sp_name_qualifier:
            kwargs['sp_name_qualifier'] = sp_name_qualifier
        if name_qualifier:
            kwargs['name_qualifier'] = name_qualifier
        if sp_provided_id:
            kwargs['sp_provided_id'] = sp_provided_id
        res = []
        for item in self.mdb.get(userid, **kwargs):
            res.append(from_dict(item['name_id'], ONTS, True))
        return res

    def find_local_id(self, name_id):
        cnid = to_dict(name_id, MMODS, True)
        for item in self.mdb.get(name_id=cnid):
            return item[self.mdb.primary_key]
        return None

    def match_local_id(self, userid, sp_name_qualifier, name_qualifier):
        """
        Match a local persistent identifier.

        Look for an existing persistent NameID matching userid,
        sp_name_qualifier and name_qualifier.
        """
        filter = {'name_id.sp_name_qualifier': sp_name_qualifier, 'name_id.name_qualifier': name_qualifier, 'name_id.format': NAMEID_FORMAT_PERSISTENT}
        res = self.mdb.get(value=userid, **filter)
        if not res:
            return None
        return from_dict(res[0]['name_id'], ONTS, True)

    def remove_remote(self, name_id):
        cnid = to_dict(name_id, MMODS, True)
        self.mdb.remove(name_id=cnid)

    def handle_name_id_mapping_request(self, name_id, name_id_policy):
        _id = self.find_local_id(name_id)
        if not _id:
            raise Unknown('Unknown entity')
        if name_id_policy.allow_create == 'false':
            raise PolicyError('Not allowed to create new identifier')
        return self.construct_nameid(_id, name_id_policy=name_id_policy)