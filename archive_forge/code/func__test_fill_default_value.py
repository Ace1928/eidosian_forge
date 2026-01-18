from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def _test_fill_default_value(self, attr_inst, expected, res_dict, check_allow_post=True):
    attr_inst.fill_post_defaults(res_dict, check_allow_post=check_allow_post)
    self.assertEqual(expected, res_dict)