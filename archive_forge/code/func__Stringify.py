from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _Stringify(value):
    """Represents value as a JSON string if it's not a string."""
    if value is None:
        return ''
    elif isinstance(value, console_attr.Colorizer):
        return value
    elif isinstance(value, six.string_types):
        return console_attr.Decode(value)
    elif isinstance(value, float):
        return resource_transform.TransformFloat(value)
    elif hasattr(value, '__str__'):
        return six.text_type(value)
    else:
        return json.dumps(value, sort_keys=True)