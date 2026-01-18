from oslo_utils import netutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def _validate_segment_update_supported(self):
    if self.properties[self.SEGMENT] is not None:
        msg = _('Updating the subnet segment assciation only allowed when the current segment_id is None. The subnet is currently associated with segment. In this state update')
        raise exception.ResourceActionNotSupported(action=msg)
    else:
        return True