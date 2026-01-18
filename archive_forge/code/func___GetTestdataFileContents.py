import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def __GetTestdataFileContents(self, filename):
    file_path = os.path.join(os.path.dirname(__file__), self._TESTDATA_PREFIX, filename)
    file_contents = open(file_path).read()
    self.assertIsNotNone(file_contents, msg='Could not read file %s' % filename)
    return file_contents