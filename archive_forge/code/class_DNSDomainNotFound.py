from neutron_lib._i18n import _
from neutron_lib import exceptions
class DNSDomainNotFound(exceptions.NotFound):
    message = _('Domain %(dns_domain)s not found in the external DNS service')