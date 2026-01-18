import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
def _cleanupParsedProperties(self):
    super(EC2ResponseError, self)._cleanupParsedProperties()
    self._errorResultSet = []
    for p in 'errors':
        setattr(self, p, None)