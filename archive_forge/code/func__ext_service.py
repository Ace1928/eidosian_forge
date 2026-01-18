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
def _ext_service(self, entity_id, typ, service, binding):
    try:
        srvs = self[entity_id][typ]
    except KeyError:
        return None
    if not srvs:
        return srvs
    res = []
    for srv in srvs:
        if 'extensions' in srv:
            for elem in srv['extensions']['extension_elements']:
                if elem['__class__'] == service:
                    if elem['binding'] == binding:
                        res.append(elem)
    return res