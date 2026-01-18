import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
class TestDict2Tuples(base.BaseTestCase):

    def test_dict(self):
        input_dict = {'foo': 'bar', '42': 'baz', 'aaa': 'zzz'}
        expected = (('42', 'baz'), ('aaa', 'zzz'), ('foo', 'bar'))
        output_tuple = helpers.dict2tuple(input_dict)
        self.assertEqual(expected, output_tuple)