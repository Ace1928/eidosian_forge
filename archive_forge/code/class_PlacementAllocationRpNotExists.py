from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementAllocationRpNotExists(exceptions.BadRequest):
    message = _('Resource provider %(resource_provider)s for %(consumer)s does not exist')