from oslo_utils import excutils
from neutron_lib._i18n import _
class IllegalSubnetPoolIpVersionAssociationToAddressScope(BadRequest):
    message = _('Illegal subnetpool association: subnetpool %(subnetpool_id)s cannot associate with address scope %(address_scope_id)s because subnetpool ip_version is not %(ip_version)s.')