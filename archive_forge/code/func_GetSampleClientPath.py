import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
def GetSampleClientPath(api_name, *path):
    return os.path.join(os.path.dirname(__file__), api_name + '_sample', *path)