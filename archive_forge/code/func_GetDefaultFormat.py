from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
import six
def GetDefaultFormat(predictions):
    if not isinstance(predictions, list):
        return 'json'
    elif not predictions:
        return None
    elif isinstance(predictions[0], dict):
        keys = ', '.join(sorted(predictions[0].keys()))
        return '\n          table(\n              predictions:format="table(\n                  {}\n              )"\n          )'.format(keys)
    else:
        return 'table[no-heading](predictions)'