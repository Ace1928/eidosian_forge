from oslo_utils import excutils
from neutron_lib._i18n import _
class StateInvalid(BadRequest):
    message = _('Unsupported port state: %(port_state)s.')