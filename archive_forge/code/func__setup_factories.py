from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def _setup_factories(self, extrascopes, **kw):
    for factory, (scope, Default) in {'response_factory': (boto.mws.response, self.ResponseFactory), 'response_error_factory': (boto.mws.exception, self.ResponseErrorFactory)}.items():
        if factory in kw:
            setattr(self, '_' + factory, kw.pop(factory))
        else:
            scopes = extrascopes + [scope]
            setattr(self, '_' + factory, Default(scopes=scopes))
    return kw