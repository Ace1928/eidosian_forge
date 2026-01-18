from neutron_lib._i18n import _
from neutron_lib import exceptions
class SubnetInUseByVPNService(exceptions.InUse):
    message = _('Subnet %(subnet_id)s is used by VPNService %(vpnservice_id)s')