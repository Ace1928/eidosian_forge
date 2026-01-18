import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class S3CopyError(StorageCopyError):
    """
    Error copying a key on S3.
    """
    pass