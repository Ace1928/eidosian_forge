from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
import six
def ParseCreateFaultFromYaml(fault, fault_config, parent):
    """Converts the given fault dict to the corresponding import request.

  Args:
    fault: faultId, fault name
    fault_config: dict, fault configuation of the create fault request.
    parent: parent for fault resource

  Returns:
    FaultinjectiontestingProjectsLocationsFaultsCreateRequest
  Raises:
    InvalidFaultConfigurationError: If the fault config is invalid.
  """
    messages = GetMessagesModule(release_track=base.ReleaseTrack.ALPHA)
    request = messages.FaultinjectiontestingProjectsLocationsFaultsCreateRequest
    try:
        import_request_message = encoding.DictToMessage({'fault': fault_config, 'faultId': fault, 'parent': parent}, request)
    except AttributeError:
        raise InvalidFaultConfigurationError('An error occurred while parsing the serialized fault. Please check your input file.')
    unrecognized_field_paths = _GetUnrecognizedFieldPaths(import_request_message)
    if unrecognized_field_paths:
        error_msg_lines = ['Invalid fault config, the following fields are ' + 'unrecognized:']
        error_msg_lines += unrecognized_field_paths
        raise InvalidFaultConfigurationError('\n'.join(error_msg_lines))
    return import_request_message