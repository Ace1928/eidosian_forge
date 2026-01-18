from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
class RepeatedField(SchemaField):
    """Represents a list of nested elements. Each item in the list is copied.

  The type of each element in the list is specified in the constructor.

  Expected input type: List
  Output type: List
  """

    def __init__(self, target_name=None, converter=None, element=None):
        """Constructor.

    Args:
      target_name: New field name to use when creating an output dictionary. If
        None is specified, then the original name is used.
      converter: A function which performs a transformation on the value of the
        field.
      element: A SchemaField element defining the type of every element in the
        list. The input structure is expected to be homogenous.

    Raises:
      ValueError: If an element has not been specified or if the element type is
      incompatible with a repeated field.
    """
        super(RepeatedField, self).__init__(target_name, converter)
        self.element = element
        if not self.element:
            raise ValueError('Element required for a repeated field')
        if isinstance(self.element, Map):
            raise ValueError('Repeated maps are not supported')

    def _VisitInternal(self, value):
        ValidateType(value, list)
        result = []
        for item in value:
            result.append(self.element.ConvertValue(item))
        return result