from oslo_utils import excutils
from neutron_lib._i18n import _
class MaxPrefixSubnetAllocationError(BadRequest):
    message = _('Unable to allocate subnet with prefix length %(prefixlen)s, maximum allowed prefix is %(max_prefixlen)s.')