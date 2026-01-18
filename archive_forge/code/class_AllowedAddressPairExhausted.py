from neutron_lib._i18n import _
from neutron_lib import exceptions
class AllowedAddressPairExhausted(exceptions.BadRequest):
    message = _('The number of allowed address pair exceeds the maximum %(quota)s.')