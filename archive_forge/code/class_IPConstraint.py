import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class IPConstraint(constraints.BaseCustomConstraint):

    def validate(self, value, context, template=None):
        self._error_message = 'Invalid IP address'
        if not isinstance(value, str):
            return False
        msg = validators.validate_ip_address(value)
        if msg is not None:
            return False
        return True