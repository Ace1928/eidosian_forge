from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecPolicyInUse(exceptions.InUse):
    message = _("IPsecPolicy %(ipsecpolicy_id)s is in use by existing IPsecSiteConnection and can't be updated or deleted")