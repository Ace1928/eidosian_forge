import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestReversedEnumerate(testscenarios.TestWithScenarios, test.TestCase):
    scenarios = [('ten', {'sample': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}), ('empty', {'sample': []}), ('negative', {'sample': [-1, -2, -3]}), ('one', {'sample': [1]}), ('abc', {'sample': ['a', 'b', 'c']}), ('ascii_letters', {'sample': list(string.ascii_letters)})]

    def test_sample_equivalence(self):
        expected = list(reversed(list(enumerate(self.sample))))
        actual = list(misc.reverse_enumerate(self.sample))
        self.assertEqual(expected, actual)