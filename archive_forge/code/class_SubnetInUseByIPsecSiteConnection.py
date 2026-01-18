from neutron_lib._i18n import _
from neutron_lib import exceptions
class SubnetInUseByIPsecSiteConnection(exceptions.InUse):
    message = _('Subnet %(subnet_id)s is used by ipsec site connection %(ipsec_site_connection_id)s')