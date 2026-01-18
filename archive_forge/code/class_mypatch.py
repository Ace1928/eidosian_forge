import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
class mypatch(_patch):

    def stop(self):
        stopped.append(attribute)
        return super(mypatch, self).stop()