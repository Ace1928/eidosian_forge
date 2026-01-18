import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class StorageCreateError(BotoServerError):
    """
    Error creating a bucket or key on a storage service.
    """

    def __init__(self, status, reason, body=None):
        self.bucket = None
        super(StorageCreateError, self).__init__(status, reason, body)

    def endElement(self, name, value, connection):
        if name == 'BucketName':
            self.bucket = value
        else:
            return super(StorageCreateError, self).endElement(name, value, connection)