import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class DNSDomainConstraint(DNSNameConstraint):

    def validate(self, value, context):
        if not value:
            return True
        if not super(DNSDomainConstraint, self).validate(value, context):
            return False
        if not value.endswith('.'):
            self._error_message = "'%s' must end with '.'." % value
            return False
        return True