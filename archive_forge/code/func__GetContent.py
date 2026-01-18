import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def _GetContent(file_path):
    with open(file_path) as f:
        return f.read()