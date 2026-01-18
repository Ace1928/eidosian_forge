from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def ParamSpecTransform(param_spec):
    if 'default' in param_spec:
        param_spec['default'] = ParamValueTransform(param_spec['default'])
    _ConvertToUpperCase(param_spec, 'type')