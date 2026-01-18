from oslo_utils import excutils
from neutron_lib._i18n import _
class MinPrefixSubnetAllocationError(BadRequest):
    message = _('Unable to allocate subnet with prefix length %(prefixlen)s, minimum allowed prefix is %(min_prefixlen)s.')