import os
import datetime
import boto.utils
from boto.compat import json
class AssumedRole(object):
    """
    :ivar user: The assumed role user.
    :ivar credentials: A Credentials object containing the credentials.
    """

    def __init__(self, connection=None, credentials=None, user=None):
        self._connection = connection
        self.credentials = credentials
        self.user = user

    def startElement(self, name, attrs, connection):
        if name == 'Credentials':
            self.credentials = Credentials()
            return self.credentials
        elif name == 'AssumedRoleUser':
            self.user = User()
            return self.user

    def endElement(self, name, value, connection):
        pass