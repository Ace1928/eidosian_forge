from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkIdOrRouterIdRequiredError(NeutronException):
    message = _('Both network_id and router_id are None. One must be provided.')