from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def _ConvertToUpperCase(input_map: MutableMapping[str, str], key: str):
    if key in input_map:
        input_map[key] = input_map[key].upper()