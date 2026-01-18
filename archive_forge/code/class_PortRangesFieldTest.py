import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
class PortRangesFieldTest(test_base.BaseTestCase, TestField):

    def setUp(self):
        super(PortRangesFieldTest, self).setUp()
        self.field = common_types.PortRangesField()
        self.coerce_good_values = [(val, val) for val in ('80:80', '80:90', '80', 80)]
        self.coerce_bad_values = ('x', 0, 99999, '99999', '9999:', ':9999', '99999:100000', '80:70')
        self.to_primitive_values = self.coerce_good_values
        self.from_primitive_values = self.coerce_good_values