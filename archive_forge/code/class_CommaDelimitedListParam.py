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
class CommaDelimitedListParam(ParsedParameter, collections.abc.Sequence):
    """A template parameter of type "CommaDelimitedList"."""
    __slots__ = tuple()

    def default_parsed(self):
        return []

    def parse(self, value):
        if isinstance(value, list):
            return [str(x) for x in value]
        try:
            return param_utils.delim_string_to_list(value)
        except (KeyError, AttributeError) as err:
            message = _('Value must be a comma-delimited list string: %s')
            raise ValueError(message % str(err))
        return value

    def value(self):
        if self.has_value():
            return self.parsed
        raise exception.UserParameterMissing(key=self.name)

    def __len__(self):
        """Return the length of the list."""
        return len(self.parsed)

    def __getitem__(self, index):
        """Return an item from the list."""
        return self.parsed[index]

    @classmethod
    def _value_as_text(cls, value):
        return ','.join(value)

    def _validate(self, val, context):
        try:
            parsed = self.parse(val)
        except ValueError as ex:
            raise exception.StackValidationFailed(message=str(ex))
        self.schema.validate_value(parsed, context)