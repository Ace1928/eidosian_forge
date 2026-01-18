from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def ParamValueTransform(param_value):
    if isinstance(param_value, str) or isinstance(param_value, float) or isinstance(param_value, int):
        return {'type': 'STRING', 'stringVal': str(param_value)}
    elif isinstance(param_value, list):
        return {'type': 'ARRAY', 'arrayVal': param_value}
    else:
        raise cloudbuild_exceptions.InvalidYamlError('Unsupported param value type. {msg_type}'.format(msg_type=type(param_value)))