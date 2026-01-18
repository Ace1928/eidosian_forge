from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterNotFoundInRouterFactory(exceptions.NeutronException):
    message = _("Router '%(router_id)s' with features '%(features)s' could not be found in the router factory.")