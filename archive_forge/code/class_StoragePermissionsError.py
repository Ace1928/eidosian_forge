import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class StoragePermissionsError(BotoClientError):
    """
    Permissions error when accessing a bucket or key on a storage service.
    """
    pass