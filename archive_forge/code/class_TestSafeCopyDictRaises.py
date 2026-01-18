import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestSafeCopyDictRaises(testscenarios.TestWithScenarios):
    scenarios = [('list', {'original': [1, 2], 'exception': TypeError}), ('tuple', {'original': (1, 2), 'exception': TypeError}), ('set', {'original': set([1, 2]), 'exception': TypeError})]

    def test_exceptions(self):
        self.assertRaises(self.exception, misc.safe_copy_dict, self.original)