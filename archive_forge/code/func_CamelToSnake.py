from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def CamelToSnake(data):
    return re.sub(pattern='([A-Z]+)', repl='_\\1', string=data).lower().lstrip('_')