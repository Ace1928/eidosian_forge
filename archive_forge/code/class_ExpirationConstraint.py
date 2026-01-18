import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class ExpirationConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context):
        if not value:
            return True
        try:
            expiration_tz = timeutils.parse_isotime(value.strip())
            expiration = timeutils.normalize_time(expiration_tz)
            if expiration > timeutils.utcnow():
                return True
            raise ValueError(_('Expiration time is out of date.'))
        except Exception as ex:
            self._error_message = _('Expiration {0} is invalid: {1}').format(value, str(ex))
        return False