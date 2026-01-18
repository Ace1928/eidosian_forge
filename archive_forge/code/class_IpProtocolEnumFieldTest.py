import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class IpProtocolEnumFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(IpProtocolEnumFieldTest, self).setUp()
        self.field = common_types.IpProtocolEnumField()
        self.coerce_good_values = [(val, val) for val in itertools.chain(const.IP_PROTOCOL_MAP.keys(), [str(v) for v in range(256)])]
        self.coerce_bad_values = ['test', 'Udp', 256]
        self.to_primitive_values = self.coerce_good_values
        self.from_primitive_values = self.coerce_good_values

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual("'%s'" % in_val, self.field.stringify(in_val))