import hashlib
import math
import binascii
from boto.compat import six
class ResettingFileSender(object):

    def __init__(self, archive):
        self._archive = archive
        self._starting_offset = archive.tell()

    def __call__(self, connection, method, path, body, headers):
        try:
            connection.request(method, path, self._archive, headers)
            return connection.getresponse()
        finally:
            self._archive.seek(self._starting_offset)