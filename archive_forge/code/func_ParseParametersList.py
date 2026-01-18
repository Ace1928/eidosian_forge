from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.dataflow import exceptions
from googlecloudsdk.core.util import files
import six
def ParseParametersList(parameters):
    """Parses a list of parameters.

  Arguments:
    parameters: A list of parameter strings with the format name:type:value,
      for example min_word_count:INT64:250.

  Returns:
    A JSON string containing the parameters.
  """
    results = []
    for parameter in parameters:
        results.append(_ParseParameter(parameter))
    return json.dumps(results)