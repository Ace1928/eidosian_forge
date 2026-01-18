import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class _EC2Error(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.error_code = None
        self.error_message = None

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Code':
            self.error_code = value
        elif name == 'Message':
            self.error_message = value
        else:
            return None