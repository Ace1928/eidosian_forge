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
def _validate_port(self, attr, value):
    if self._is_port_acceptable(value):
        return
    raise ValueError(_('The port %(value)s does not respect the range (%(min)s, %(max)s) in field %(attr)s') % {'attr': attr, 'value': value, 'min': lib_constants.PORT_RANGE_MIN, 'max': lib_constants.PORT_RANGE_MAX})