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
@staticmethod
def _prop_def_from_prop(name, schema):
    """Return a provider template property definition for a property."""
    if schema.type == Schema.LIST:
        return {'Fn::Split': [',', {'Ref': name}]}
    else:
        return {'Ref': name}