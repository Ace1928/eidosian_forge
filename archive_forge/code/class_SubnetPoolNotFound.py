from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetPoolNotFound(NotFound):
    message = _('Subnet pool %(subnetpool_id)s could not be found.')