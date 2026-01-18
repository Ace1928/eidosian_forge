from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecPolicyNotFound(exceptions.NotFound):
    message = _('IPsecPolicy %(ipsecpolicy_id)s could not be found')