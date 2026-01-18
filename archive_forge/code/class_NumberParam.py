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
class NumberParam(Parameter):
    """A template parameter of type "Number"."""
    __slots__ = tuple()

    def __int__(self):
        """Return an integer representation of the parameter."""
        return int(super(NumberParam, self).value())

    def __float__(self):
        """Return a float representation of the parameter."""
        return float(super(NumberParam, self).value())

    def _validate(self, val, context):
        try:
            Schema.str_to_num(val)
        except (ValueError, TypeError) as ex:
            raise exception.StackValidationFailed(message=str(ex))
        self.schema.validate_value(val, context)

    def value(self):
        return Schema.str_to_num(super(NumberParam, self).value())