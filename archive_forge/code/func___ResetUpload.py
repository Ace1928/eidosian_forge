import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def __ResetUpload(self, size, auto_transfer=True):
    self.__content = ''.join((random.choice(string.ascii_letters) for _ in range(size)))
    self.__buffer = six.StringIO(self.__content)
    self.__upload = storage.Upload.FromStream(self.__buffer, 'text/plain', auto_transfer=auto_transfer)