from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
def _LabelsByKey(self):
    """Returns an object that maps keys to projection labels.

    Returns:
      An object of keys to projection labels, None if all labels are empty.
    """
    labels = {}
    for c in self.column_attributes.Columns():
        key_name = resource_lex.GetKeyName(c.key)
        labels[key_name] = c.attribute.label
    return labels if any(labels) else None