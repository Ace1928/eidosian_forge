from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def ResultValueTransform(result_value):
    """Transforms the string result value from Tekton to GCB resultValue struct."""
    if isinstance(result_value, str) or isinstance(result_value, float) or isinstance(result_value, int):
        return {'type': 'STRING', 'stringVal': str(result_value)}
    elif isinstance(result_value, list):
        return {'type': 'ARRAY', 'arrayVal': result_value}
    elif isinstance(result_value, object):
        return {'type': 'OBJECT', 'objectVal': result_value}
    else:
        raise cloudbuild_exceptions.InvalidYamlError('Unsupported param value type. {msg_type}'.format(msg_type=type(result_value)))