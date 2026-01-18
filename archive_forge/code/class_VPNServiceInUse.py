from neutron_lib._i18n import _
from neutron_lib import exceptions
class VPNServiceInUse(exceptions.InUse):
    message = _('VPNService %(vpnservice_id)s is still in use')