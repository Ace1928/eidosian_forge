from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def ParamDictTransform(params):
    for param in params:
        param['value'] = ParamValueTransform(param['value'])