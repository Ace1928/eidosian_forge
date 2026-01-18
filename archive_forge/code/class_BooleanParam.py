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
class BooleanParam(Parameter):
    """A template parameter of type "Boolean"."""
    __slots__ = tuple()

    def _validate(self, val, context):
        try:
            strutils.bool_from_string(val, strict=True)
        except ValueError as ex:
            raise exception.StackValidationFailed(message=str(ex))
        self.schema.validate_value(val, context)

    def value(self):
        if self.user_value is not None:
            raw_value = self.user_value
        else:
            raw_value = self.default()
        return strutils.bool_from_string(str(raw_value), strict=True)