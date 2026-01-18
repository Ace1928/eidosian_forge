from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupNotFound(exceptions.NotFound):
    message = _('Firewall group %(firewall_id)s could not be found.')