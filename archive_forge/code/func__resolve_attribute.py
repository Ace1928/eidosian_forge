from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _resolve_attribute(self, name):
    if self.resource_id is None:
        return None
    resource = self._show_resource()
    return resource[name]