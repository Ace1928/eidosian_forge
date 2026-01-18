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
class PortRanges(obj_fields.FieldType):

    @staticmethod
    def _is_port_acceptable(port):
        start = lib_constants.PORT_RANGE_MIN
        end = lib_constants.PORT_RANGE_MAX
        return start <= port <= end

    def get_schema(self):
        return {'type': ['string', 'integer']}

    def _validate_port(self, attr, value):
        if self._is_port_acceptable(value):
            return
        raise ValueError(_('The port %(value)s does not respect the range (%(min)s, %(max)s) in field %(attr)s') % {'attr': attr, 'value': value, 'min': lib_constants.PORT_RANGE_MIN, 'max': lib_constants.PORT_RANGE_MAX})

    def coerce(self, obj, attr, value):
        if isinstance(value, int):
            self._validate_port(attr, value)
            return value
        if isinstance(value, str):
            if value.isnumeric():
                self._validate_port(attr, int(value))
                return value
            values = value.split(':')
            if len(values) == 2:
                start, end = list(map(int, values))
                if start > end:
                    raise ValueError(_('The first port %(start)s must be less or equals than the second port %(end)s of the port range configuration %(value)sin field %(attr)s.') % {'attr': attr, 'value': value, 'start': start, 'end': end})
                self._validate_port(attr, start)
                self._validate_port(attr, end)
                return value
            raise ValueError(_('The field %(attr)s must be in the format PORT_RANGE orPORT_RANGE:PORT_RANGE (two numeric values separated by a colon), and PORT_RANGE must be a numeric value and respect the range (%(min)s, %(max)s).') % {'attr': attr, 'min': lib_constants.PORT_RANGE_MIN, 'max': lib_constants.PORT_RANGE_MAX})
        raise ValueError(_('An string/int PORT_RANGE or a string with PORT_RANGE:PORT_RANGE format is expected in field %(attr)s, not a %(type)s') % {'attr': attr, 'type': value})