import copy
from unittest import mock
from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack import resource as sdk_resource
from osc_lib.utils import columns as column_util
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
class BgpvpnFakeAssoc(object):
    _assoc_res_name = 'fake_resource'
    _resource = '%s_association' % _assoc_res_name
    _resource_plural = '%ss' % _resource
    _attr_map = (('id', 'ID', column_util.LIST_BOTH), ('%s_id' % _assoc_res_name, '%s ID' % _assoc_res_name.capitalize(), column_util.LIST_BOTH), ('name', 'Name', column_util.LIST_BOTH), ('project_id', 'Project ID', column_util.LIST_BOTH))
    _formatters = {}