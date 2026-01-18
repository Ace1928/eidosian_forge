from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementAggregateNotFound(exceptions.NotFound):
    message = _('Aggregate not found for resource provider %(resource_provider)s.')