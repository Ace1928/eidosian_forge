from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def _core_resource_attributes():
    resources = {}
    for core_def in [network.RESOURCE_ATTRIBUTE_MAP, port.RESOURCE_ATTRIBUTE_MAP, subnet.RESOURCE_ATTRIBUTE_MAP, subnetpool.RESOURCE_ATTRIBUTE_MAP]:
        resources.update(core_def)
    return resources