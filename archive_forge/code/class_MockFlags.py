import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
class MockFlags(object):
    auth_host_name = HOST
    auth_host_port = PORT
    auth_local_webserver = False