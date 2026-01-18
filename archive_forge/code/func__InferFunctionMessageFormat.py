from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.eventarc import types as trigger_types
def _InferFunctionMessageFormat(data, undefined='-'):
    """Returns Cloud Functions product version.

  Infers data type by checking whether the object contains particular fields of
  CloudFunction (1st Gen Function message type) or Function (2nd Gen Function
  message type). Notes that Function can be used for both 1st Gen and 2nd Gen
  functions.

  Args:
    data: JSON-serializable 1st and 2nd gen Functions objects.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing inferred product version.
  """
    entry_point = data.get('entryPoint')
    build_id = data.get('buildId')
    runtime = data.get('runtime')
    if any([entry_point, build_id, runtime]):
        return CLOUD_FUNCTION
    build_config = data.get('buildConfig')
    service_config = data.get('serviceConfig')
    if any([build_config, service_config]):
        return FUNCTION
    return undefined