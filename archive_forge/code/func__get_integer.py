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
def _get_integer(self, value):
    if value is None:
        value = self.has_default() and self.default() or 0
    try:
        value = int(value)
    except ValueError:
        raise TypeError(_("Value '%s' is not an integer") % value)
    else:
        return value