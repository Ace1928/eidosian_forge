from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class PropertyFallthrough(_FallthroughBase):
    """Gets an attribute from a property."""

    def __init__(self, prop, plural=False):
        """Initializes a fallthrough for the property associated with the attribute.

    Args:
      prop: googlecloudsdk.core.properties._Property, a property.
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
        resource argument is plural (i.e. parses to a list).
    """
        hint = 'set the property `{}`'.format(prop)
        super(PropertyFallthrough, self).__init__(hint, plural=plural)
        self.property = prop

    def _Call(self, parsed_args):
        del parsed_args
        try:
            return self.property.GetOrFail()
        except (properties.InvalidValueError, properties.RequiredPropertyError):
            return None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return other.property == self.property

    def __hash__(self):
        return hash(self.property)