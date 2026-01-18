from neutron_lib._i18n import _
from neutron_lib import exceptions
class NoProviderFoundForFlavor(exceptions.NotFound):
    message = _('No service provider found for flavor %(flavor_id)s')