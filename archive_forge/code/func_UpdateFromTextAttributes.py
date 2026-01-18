from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console.style import mappings
from googlecloudsdk.core.console.style import text
import six
def UpdateFromTextAttributes(self, text_attributes):
    if not text_attributes:
        return self
    new_color = text_attributes.color or self.color
    new_attrs = getattr(text_attributes, 'attrs', []) + self.attrs
    return self.__class__(new_color, new_attrs)