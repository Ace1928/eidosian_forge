import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def _get_string(self, value):
    if value is None:
        value = self.has_default() and self.default() or ''
    if not isinstance(value, str):
        if isinstance(value, (bool, int)):
            value = str(value)
        else:
            raise ValueError(_('Value must be a string; got %r') % value)
    return value