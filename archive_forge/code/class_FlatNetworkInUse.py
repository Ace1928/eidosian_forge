from oslo_utils import excutils
from neutron_lib._i18n import _
class FlatNetworkInUse(InUse):
    message = _('Unable to create the flat network. Physical network %(physical_network)s is in use.')