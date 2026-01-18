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
def _param_def_from_prop(schema):
    """Return a template parameter definition corresponding to property."""
    param_type_map = {schema.INTEGER: parameters.Schema.NUMBER, schema.STRING: parameters.Schema.STRING, schema.NUMBER: parameters.Schema.NUMBER, schema.BOOLEAN: parameters.Schema.BOOLEAN, schema.MAP: parameters.Schema.MAP, schema.LIST: parameters.Schema.LIST}

    def param_items():
        yield (parameters.TYPE, param_type_map[schema.type])
        if schema.description is not None:
            yield (parameters.DESCRIPTION, schema.description)
        if schema.default is not None:
            yield (parameters.DEFAULT, schema.default)
        for constraint in schema.constraints:
            if isinstance(constraint, constr.Length):
                if constraint.min is not None:
                    yield (parameters.MIN_LENGTH, constraint.min)
                if constraint.max is not None:
                    yield (parameters.MAX_LENGTH, constraint.max)
            elif isinstance(constraint, constr.Range):
                if constraint.min is not None:
                    yield (parameters.MIN_VALUE, constraint.min)
                if constraint.max is not None:
                    yield (parameters.MAX_VALUE, constraint.max)
            elif isinstance(constraint, constr.AllowedValues):
                yield (parameters.ALLOWED_VALUES, list(constraint.allowed))
            elif isinstance(constraint, constr.AllowedPattern):
                yield (parameters.ALLOWED_PATTERN, constraint.pattern)
        if schema.type == schema.BOOLEAN:
            yield (parameters.ALLOWED_VALUES, ['True', 'true', 'False', 'false'])
    return dict(param_items())