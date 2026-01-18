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
class JsonParam(ParsedParameter):
    """A template parameter who's value is map or list."""
    __slots__ = tuple()

    def default_parsed(self):
        return {}

    def parse(self, value):
        try:
            val = value
            if not isinstance(val, str):
                val = jsonutils.dumps(val, default=None)
            if val:
                return jsonutils.loads(val)
        except (ValueError, TypeError) as err:
            message = _('Value must be valid JSON: %s') % err
            raise ValueError(message)
        return value

    def value(self):
        if self.has_value():
            return self.parsed
        raise exception.UserParameterMissing(key=self.name)

    def __getitem__(self, key):
        return self.parsed[key]

    def __iter__(self):
        return iter(self.parsed)

    def __len__(self):
        return len(self.parsed)

    @classmethod
    def _value_as_text(cls, value):
        return encodeutils.safe_decode(jsonutils.dumps(value))

    def _validate(self, val, context):
        try:
            parsed = self.parse(val)
        except ValueError as ex:
            raise exception.StackValidationFailed(message=str(ex))
        self.schema.validate_value(parsed, context)