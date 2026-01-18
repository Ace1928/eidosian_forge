from neutron_lib._i18n import _
from neutron_lib import exceptions
class PlacementResourceClassNotFound(exceptions.NotFound):
    message = _('Placement resource class not found %(resource_class)s')