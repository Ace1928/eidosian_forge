from heat.common import exception
from heat.common.i18n import _
from heat.common import netutils
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _validate_prefix_bounds(self):
    min_prefixlen = self.properties[self.MIN_PREFIXLEN]
    default_prefixlen = self.properties[self.DEFAULT_PREFIXLEN]
    max_prefixlen = self.properties[self.MAX_PREFIXLEN]
    msg_fmt = _('Illegal prefix bounds: %(key1)s=%(value1)s, %(key2)s=%(value2)s.')
    if min_prefixlen and max_prefixlen and (min_prefixlen > max_prefixlen):
        msg = msg_fmt % dict(key1=self.MAX_PREFIXLEN, value1=max_prefixlen, key2=self.MIN_PREFIXLEN, value2=min_prefixlen)
        raise exception.StackValidationFailed(message=msg)
    if default_prefixlen:
        if max_prefixlen and default_prefixlen > max_prefixlen:
            msg = msg_fmt % dict(key1=self.MAX_PREFIXLEN, value1=max_prefixlen, key2=self.DEFAULT_PREFIXLEN, value2=default_prefixlen)
            raise exception.StackValidationFailed(message=msg)
        if min_prefixlen and min_prefixlen > default_prefixlen:
            msg = msg_fmt % dict(key1=self.MIN_PREFIXLEN, value1=min_prefixlen, key2=self.DEFAULT_PREFIXLEN, value2=default_prefixlen)
            raise exception.StackValidationFailed(message=msg)