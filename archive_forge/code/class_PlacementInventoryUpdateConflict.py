from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementInventoryUpdateConflict(exceptions.Conflict):
    message = _('Placement inventory update conflict for resource provider %(resource_provider)s, resource class %(resource_class)s.')