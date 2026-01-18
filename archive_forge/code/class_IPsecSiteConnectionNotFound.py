from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecSiteConnectionNotFound(exceptions.NotFound):
    message = _('ipsec_site_connection %(ipsec_site_conn_id)s not found')