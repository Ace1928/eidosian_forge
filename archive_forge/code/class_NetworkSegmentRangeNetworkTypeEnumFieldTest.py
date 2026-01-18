import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class NetworkSegmentRangeNetworkTypeEnumFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(NetworkSegmentRangeNetworkTypeEnumFieldTest, self).setUp()
        self.field = common_types.NetworkSegmentRangeNetworkTypeEnumField()
        self.coerce_good_values = [(val, val) for val in [const.TYPE_VLAN, const.TYPE_VXLAN, const.TYPE_GRE, const.TYPE_GENEVE]]
        self.coerce_bad_values = [const.TYPE_FLAT, 'foo-network-type']
        self.to_primitive_values = self.coerce_good_values
        self.from_primitive_values = self.coerce_good_values

    def test_stringify(self):
        for in_val, out_val in self.coerce_good_values:
            self.assertEqual("'%s'" % in_val, self.field.stringify(in_val))