import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class InvalidCorsError(Exception):
    """Exception raised when CORS XML is invalid."""

    def __init__(self, message):
        super(InvalidCorsError, self).__init__(message)
        self.message = message