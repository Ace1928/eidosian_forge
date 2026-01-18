from neutron_lib._i18n import _
from neutron_lib import exceptions
class HANetworkCIDRNotValid(exceptions.NeutronException):
    message = _("The HA Network CIDR specified in the configuration file isn't valid; %(cidr)s.")