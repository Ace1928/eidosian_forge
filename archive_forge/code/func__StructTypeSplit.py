from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def _StructTypeSplit(type_string):
    """Yields single field-name, sub-types tuple from a StructType string."""
    while type_string:
        next_span = type_string.split(',', 1)[0]
        if '<' in next_span:
            angle_count = 0
            i = 0
            for i in range(next_span.find('<'), len(type_string)):
                if type_string[i] == '<':
                    angle_count += 1
                if type_string[i] == '>':
                    angle_count -= 1
                if angle_count == 0:
                    break
            if angle_count != 0:
                raise exceptions.Error('Malformatted struct type')
            next_span = type_string[:i + 1]
        type_string = type_string[len(next_span) + 1:]
        splits = next_span.split(None, 1)
        if len(splits) != 2:
            raise exceptions.Error('Struct parameter missing name for field')
        yield splits