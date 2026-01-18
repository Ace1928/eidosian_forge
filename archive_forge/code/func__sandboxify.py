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
def _sandboxify(self, path):
    if not self._sandboxed:
        return path
    splat = path.split('/')
    splat[-2] += '_Sandbox'
    return '/'.join(splat)