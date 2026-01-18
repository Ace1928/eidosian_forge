from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _ParseParameterType(type_string):
    """Parse a parameter type string into a JSON dict for the DF SQL launcher."""
    type_dict = {'type': type_string.upper()}
    if type_string.upper().startswith('ARRAY<') and type_string.endswith('>'):
        type_dict = collections.OrderedDict([('arrayType', _ParseParameterType(type_string[6:-1])), ('type', 'ARRAY')])
    if type_string.startswith('STRUCT<') and type_string.endswith('>'):
        type_dict = collections.OrderedDict([('structTypes', _ParseStructType(type_string[7:-1])), ('type', 'STRUCT')])
    if not type_string:
        raise exceptions.Error('Query parameter missing type')
    return type_dict