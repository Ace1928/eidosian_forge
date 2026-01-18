import croniter
import eventlet
import netaddr
from neutron_lib.api import validators
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import netutils as heat_netutils
from heat.engine import constraints
class RelativeDNSNameConstraint(DNSNameConstraint):

    def validate(self, value, context):
        if not value:
            return True
        if value.endswith('.'):
            self._error_message = _("'%s' is a FQDN. It should be a relative domain name.") % value
            return False
        length = len(value)
        if length > heat_netutils.FQDN_MAX_LEN - 3:
            self._error_message = _("'%(value)s' contains '%(length)s' characters. Adding a domain name will cause it to exceed the maximum length of a FQDN of '%(max_len)s'.") % {'value': value, 'length': length, 'max_len': heat_netutils.FQDN_MAX_LEN}
            return False
        return super(RelativeDNSNameConstraint, self).validate(value, context)