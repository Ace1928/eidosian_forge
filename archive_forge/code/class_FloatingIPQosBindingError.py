from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class FloatingIPQosBindingError(e.NeutronException):
    message = _('QoS binding for floating IP %(fip_id)s and policy %(policy_id)s could not be created: %(db_error)s.')