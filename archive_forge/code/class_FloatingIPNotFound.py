from neutron_lib._i18n import _
from neutron_lib import exceptions
class FloatingIPNotFound(exceptions.NotFound):
    message = _('Floating IP %(floatingip_id)s could not be found')