import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class InvalidAclError(Exception):
    """Exception raised when ACL XML is invalid."""

    def __init__(self, message):
        super(InvalidAclError, self).__init__(message)
        self.message = message