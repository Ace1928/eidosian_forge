from oslo_utils import excutils
from neutron_lib._i18n import _
class DhcpPortInUse(InUse):
    message = _('Port %(port_id)s is already acquired by another DHCP agent')