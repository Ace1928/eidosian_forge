from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupPortNotSupported(exceptions.Conflict):
    message = _("Port %(port_id)s is not supported by firewall driver '%(driver_name)s'.")