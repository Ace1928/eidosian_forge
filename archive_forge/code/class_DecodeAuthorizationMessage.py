import os
import datetime
import boto.utils
from boto.compat import json
class DecodeAuthorizationMessage(object):
    """
    :ivar request_id: The request ID.
    :ivar decoded_message: The decoded authorization message (may be JSON).
    """

    def __init__(self, request_id=None, decoded_message=None):
        self.request_id = request_id
        self.decoded_message = decoded_message

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'requestId':
            self.request_id = value
        elif name == 'DecodedMessage':
            self.decoded_message = value