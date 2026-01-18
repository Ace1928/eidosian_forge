from webob import exc
from neutron_lib.api.definitions import provider_net
from neutron_lib.api.validators import multiprovidernet as mp_validator
from neutron_lib import constants
from neutron_lib.tests import _base as base
def _build_segment(net_type=constants.ATTR_NOT_SPECIFIED, phy_net=constants.ATTR_NOT_SPECIFIED, seg_id=constants.ATTR_NOT_SPECIFIED):
    return {provider_net.NETWORK_TYPE: net_type, provider_net.PHYSICAL_NETWORK: phy_net, provider_net.SEGMENTATION_ID: seg_id}