import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class TimezoneConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        if not value:
            return True
        try:
            if zoneinfo:
                zoneinfo.ZoneInfo(value)
            else:
                pytz.timezone(value)
            return True
        except Exception as ex:
            self._error_message = _('Invalid timezone: %s') % str(ex)
        return False