import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class NonCallable(object):

    def __init__(self):
        pass