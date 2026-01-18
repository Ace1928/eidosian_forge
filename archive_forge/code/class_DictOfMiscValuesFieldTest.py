import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class DictOfMiscValuesFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(DictOfMiscValuesFieldTest, self).setUp()
        self.field = common_types.DictOfMiscValues
        test_dict_1 = {'a': True, 'b': 1.23, 'c': ['1', 1.23, True], 'd': {'aa': 'zz'}, 'e': '10.0.0.1'}
        test_dict_str = jsonutils.dumps(test_dict_1)
        self.coerce_good_values = [(test_dict_1, test_dict_1), (test_dict_str, test_dict_1)]
        self.coerce_bad_values = [str(test_dict_1), '{"a":}']
        self.to_primitive_values = [(test_dict_1, test_dict_str)]
        self.from_primitive_values = [(test_dict_str, test_dict_1)]

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual(jsonutils.dumps(in_val), self.field.stringify(in_val))