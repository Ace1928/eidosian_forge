from heat.common import exception
from heat.common.i18n import _
from heat.common import netutils
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _validate_prefixes_for_update(self, prop_diff):
    old_prefixes = self.properties[self.PREFIXES]
    new_prefixes = prop_diff[self.PREFIXES]
    if not netutils.is_prefix_subset(old_prefixes, new_prefixes):
        msg = _('Property %(key)s updated value %(new)s should be superset of existing value %(old)s.') % dict(key=self.PREFIXES, new=sorted(new_prefixes), old=sorted(old_prefixes))
        raise exception.StackValidationFailed(message=msg)