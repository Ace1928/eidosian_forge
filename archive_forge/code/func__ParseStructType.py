from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _ParseStructType(type_string):
    """Parse a Struct QueryParameter type into a JSON dict form."""
    subtypes = []
    for name, sub_type in _StructTypeSplit(type_string):
        entry = collections.OrderedDict([('name', name), ('type', _ParseParameterType(sub_type))])
        subtypes.append(entry)
    return subtypes