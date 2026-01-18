import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class S3PermissionsError(StoragePermissionsError):
    """
    Permissions error when accessing a bucket or key on S3.
    """
    pass