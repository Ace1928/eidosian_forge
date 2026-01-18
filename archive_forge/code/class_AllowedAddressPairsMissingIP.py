from neutron_lib._i18n import _
from neutron_lib import exceptions
class AllowedAddressPairsMissingIP(exceptions.InvalidInput):
    message = _('AllowedAddressPair must contain ip_address')