from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ValueFallthrough(_FallthroughBase):
    """Gets an attribute from a property."""

    def __init__(self, value, hint=None, active=False, plural=False):
        """Initializes a fallthrough for the property associated with the attribute.

    Args:
      value: str, Denoting the fixed value to provide to the attribute.
      hint: str, Optional, If provided, used over default help_text.
      active: bool, Optional, whether the value is specified by the user on
        the command line.
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
        resource argument is plural (i.e. parses to a list).
    """
        hint = 'The default is `{}`'.format(value) if hint is None else hint
        super(ValueFallthrough, self).__init__(hint, active=active, plural=plural)
        self.value = value

    def _Call(self, parsed_args):
        del parsed_args
        return self.value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return other.value == self.value

    def __hash__(self):
        return hash(self.value)