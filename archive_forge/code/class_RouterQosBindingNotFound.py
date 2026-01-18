from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class RouterQosBindingNotFound(e.NotFound):
    message = _('QoS binding for router %(router_id)s gateway and policy %(policy_id)s could not be found.')