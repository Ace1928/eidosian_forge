from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupCannotUpdateDefault(exceptions.InUse):
    message = _('Updating default firewall group not allowed.')