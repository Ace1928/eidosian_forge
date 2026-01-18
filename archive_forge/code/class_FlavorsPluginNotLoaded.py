from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorsPluginNotLoaded(exceptions.NotFound):
    message = _('Flavors plugin not found')