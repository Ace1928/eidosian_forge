import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class SDBResponseError(BotoServerError):
    """
    Error in responses from SDB.
    """
    pass