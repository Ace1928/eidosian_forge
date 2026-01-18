from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformEnum(r, projection, enums, inverse=False, undefined=''):
    """Returns the enums dictionary description for the resource.

  Args:
    r: A JSON-serializable object.
    projection: The parent ProjectionSpec.
    enums: The name of a message enum dictionary.
    inverse: Do inverse lookup if true.
    undefined: Returns this value if there is no matching enum description.

  Returns:
    The enums dictionary description for the resource.
  """
    inverse = GetBooleanArgValue(inverse)
    type_name = GetTypeDataName(enums, 'inverse-enum' if inverse else 'enum')
    descriptions = projection.symbols.get(type_name)
    if not descriptions and inverse:
        normal = projection.symbols.get(GetTypeDataName(enums, 'enum'))
        if normal:
            descriptions = {}
            for k, v in six.iteritems(normal):
                descriptions[v] = k
            projection.symbols[type_name] = descriptions
    return descriptions.get(r, undefined) if descriptions else undefined