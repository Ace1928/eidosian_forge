import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
@classmethod
def _validate_dict(cls, param_name, schema_dict):
    cls._check_dict(schema_dict, cls.PARAMETER_KEYS, 'parameter (%s)' % param_name)
    if cls.TYPE not in schema_dict:
        raise exception.InvalidSchemaError(message=_('Missing parameter type for parameter: %s') % param_name)
    if not isinstance(schema_dict.get(cls.TAGS, []), list):
        raise exception.InvalidSchemaError(message=_('Tags property should be a list for parameter: %s') % param_name)