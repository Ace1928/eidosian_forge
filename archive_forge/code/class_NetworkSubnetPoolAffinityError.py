from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkSubnetPoolAffinityError(BadRequest):
    message = _('Subnets hosted on the same network must be allocated from the same subnet pool.')