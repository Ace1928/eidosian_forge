from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementAllocationRemoved(exceptions.BadRequest):
    message = _('Resource allocation is deleted for consumer %(consumer)s')