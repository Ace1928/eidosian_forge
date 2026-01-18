from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecSiteConnectionPeerCidrError(exceptions.InvalidInput):
    message = _('ipsec_site_connection peer cidr %(peer_cidr)s is invalid CIDR')