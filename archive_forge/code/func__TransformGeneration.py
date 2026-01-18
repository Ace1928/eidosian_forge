from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.eventarc import types as trigger_types
def _TransformGeneration(data, undefined='-'):
    """Returns Cloud Functions product version.

  Args:
    data: JSON-serializable 1st and 2nd gen Functions objects.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing inferred product version.
  """
    environment = data.get('environment')
    if environment == 'GEN_1':
        return GEN_1
    if environment == 'GEN_2':
        return GEN_2
    data_type = _InferFunctionMessageFormat(data, undefined)
    if data_type == CLOUD_FUNCTION:
        return GEN_1
    elif data_type == FUNCTION:
        return GEN_2
    return undefined