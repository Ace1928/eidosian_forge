from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
@classmethod
def _FormatProtoPath(cls, edges, field_names):
    """Returns a string representation of a path to a proto field.

    The return value represents one or more fields in a python dictionary
    representation of a message (json/yaml) that could not be decoded into the
    message as a string. The format is a dot separated list of python like sub
    field references (name, name[index], name[name]). The final element of the
    returned dot separated path may be a comma separated list of names enclosed
    in curly braces to represent multiple subfields (see examples)

    Examples:
      o Reference to a single field that could not be decoded:
        'a.b[1].c[x].d'

      o Reference to two subfields
        'a.b[1].c[x].{d,e}'

    Args:
      edges: List of objects representing python field references
             (__str__ suitably defined.)
      field_names: List of names for subfields of the message
         that could not be decoded.

    Returns:
      A string representation of a python reference to a filed or
      fields in a message that could not be decoded as described
      above.
    """
    path = [six.text_type(edge) for edge in edges]
    if len(field_names) > 1:
        path.append('{{{}}}'.format(','.join(sorted(field_names))))
    elif field_names:
        path.append(field_names[0])
    return '.'.join(path)