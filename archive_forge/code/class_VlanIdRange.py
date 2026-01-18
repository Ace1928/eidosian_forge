import itertools
import uuid
import netaddr
from oslo_serialization import jsonutils
from oslo_versionedobjects import fields as obj_fields
from neutron_lib._i18n import _
from neutron_lib import constants as lib_constants
from neutron_lib.db import constants as lib_db_const
from neutron_lib.objects import exceptions as o_exc
from neutron_lib.utils import net as net_utils
class VlanIdRange(RangeConstrainedInteger):

    def __init__(self, **kwargs):
        super().__init__(start=lib_constants.MIN_VLAN_TAG, end=lib_constants.MAX_VLAN_TAG, **kwargs)