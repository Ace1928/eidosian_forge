from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
@six.add_metaclass(abc.ABCMeta)
class _ExprOperator(_Expr):
    """Base term (<key operator operand>) node.

  ExprOperator subclasses must define the function Apply(self, value, operand)
  that returns the result of <value> <op> <operand>.

  Attributes:
    _key: Resource object key (list of str, int and/or None values).
    _normalize: The resource value normalization function.
    _operand: The term ExprOperand operand.
    _transform: Optional key value transform calls.
    key : Property decorator for the resource object key.
  """
    _TIME_TYPES = (times.datetime.date, times.datetime.time, times.datetime.timedelta, times.datetime.tzinfo)

    def __init__(self, backend, key, operand, transform):
        super(_ExprOperator, self).__init__(backend)
        self._key = key
        self._operand = operand
        self._transform = transform
        if transform:
            self._normalize = lambda x: x
        else:
            self._normalize = self.InitializeNormalization

    def InitializeNormalization(self, value):
        """Checks the first non-empty resource value to see if it can be normalized.

    This method is called at most once on the first non-empty resource value.
    After that a new normalization method is set for the remainder of the
    resource values.

    Resource values are most likely well defined protobuf string encodings. The
    RE patterns match against those.

    Args:
      value: A resource value to normalize.

    Returns:
      The normalized value.
    """
        self._normalize = lambda x: x
        if re.match('\\d\\d\\d\\d-\\d\\d-\\d\\d[ T]\\d\\d:\\d\\d:\\d\\d', value):
            try:
                value = times.ParseDateTime(value)
                tzinfo = times.LOCAL if value.tzinfo else None
                self._operand.Initialize(self._operand.list_value or self._operand.string_value, normalize=lambda x: times.ParseDateTime(x, tzinfo=tzinfo))
                self._normalize = times.ParseDateTime
            except ValueError:
                pass
        return value

    @property
    def contains_key(self):
        return True

    @property
    def key(self):
        return self._key

    def Evaluate(self, obj):
        """Evaluate a term node.

    Args:
      obj: The resource object to evaluate.
    Returns:
      The value of the operator applied to the key value and operand.
    """
        value = resource_property.Get(obj, self._key)
        if self._transform:
            value = self._transform.Evaluate(value)
        if value and isinstance(value, (list, tuple)):
            resource_values = value
        else:
            resource_values = [value]
        values = []
        for value in resource_values:
            if value:
                try:
                    value = self._normalize(value)
                except (TypeError, ValueError):
                    pass
            values.append(value)
        if self._operand.list_value:
            operands = self._operand.list_value
        else:
            operands = [self._operand]
        for value in values:
            for operand in operands:
                if operand.numeric_value is not None:
                    try:
                        if self.Apply(float(value), operand.numeric_value):
                            return True
                        if not operand.numeric_constant:
                            continue
                    except (TypeError, ValueError):
                        pass
                if not value and isinstance(operand.string_value, self._TIME_TYPES):
                    continue
                try:
                    if self.Apply(value, operand.string_value):
                        return True
                except (AttributeError, ValueError):
                    pass
                except TypeError:
                    if value is not None and (not isinstance(value, (six.string_types, dict, list))) and self.Apply(_Stringize(value), operand.string_value):
                        return True
                    if six.PY3 and value is None and self.Apply('', operand.string_value):
                        return True
        return False

    @abc.abstractmethod
    def Apply(self, value, operand):
        """Returns the value of applying a <value> <operator> <operand> term.

    Args:
      value: The term key value.
      operand: The term operand value.

    Returns:
      The Boolean value of applying a <value> <operator> <operand> term.
    """
        pass