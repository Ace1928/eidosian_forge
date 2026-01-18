from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
def _GetPropertyValue(self, dest):
    """Returns the property value for dest.

    Args:
      dest: The resource argument dest.

    Returns:
      The property value for dest.
    """
    props = []
    if self._api:
        props.append(self._api + '/' + dest)
    props.append(dest)
    for prop in props:
        try:
            return properties.FromString(prop).Get()
        except properties.NoSuchPropertyError:
            pass
    return None