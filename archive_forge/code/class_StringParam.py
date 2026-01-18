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
class StringParam(Parameter):
    """A template parameter of type "String"."""
    __slots__ = tuple()

    def _validate(self, val, context):
        self.schema.validate_value(val, context=context)

    def value(self):
        return self.schema.to_schema_type(super(StringParam, self).value())