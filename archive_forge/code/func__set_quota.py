from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _set_quota(self, props=None):
    if props is None:
        props = self.properties
    kwargs = dict(((k, v) for k, v in props.items() if k != self.PROJECT and v is not None))
    self.client().quotas.update(props.get(self.PROJECT), **kwargs)