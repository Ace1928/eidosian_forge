from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecSiteConnectionMtuError(exceptions.InvalidInput):
    message = _('ipsec_site_connection MTU %(mtu)d is too small for ipv%(version)s')