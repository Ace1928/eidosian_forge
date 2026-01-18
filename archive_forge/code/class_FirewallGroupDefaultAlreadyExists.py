from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupDefaultAlreadyExists(exceptions.InUse):
    """Default firewall group conflict exception

    Occurs when a user creates firewall group named 'default'.
    """
    message = _("Default firewall group already exists. 'default' is the reserved name for firewall group.")