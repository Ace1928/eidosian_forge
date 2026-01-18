from unittest import mock
from oslotest import base
from oslotest import mock_fixture
from six.moves import builtins
import os
from os_win import exceptions
from os_win.utils import baseutils
class TestingException(Exception):
    pass