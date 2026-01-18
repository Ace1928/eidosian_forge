import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class StorageResponseError(BotoServerError):
    """
    Error in response from a storage service.
    """

    def __init__(self, status, reason, body=None):
        self.resource = None
        super(StorageResponseError, self).__init__(status, reason, body)

    def startElement(self, name, attrs, connection):
        return super(StorageResponseError, self).startElement(name, attrs, connection)

    def endElement(self, name, value, connection):
        if name == 'Resource':
            self.resource = value
        else:
            return super(StorageResponseError, self).endElement(name, value, connection)

    def _cleanupParsedProperties(self):
        super(StorageResponseError, self)._cleanupParsedProperties()
        for p in 'resource':
            setattr(self, p, None)