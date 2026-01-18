from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
import six
def ParseUpdateExperimentFromYaml(experiment, experiment_config):
    """Converts the given fault dict to the corresponding import request.

  Args:
    experiment: experimentId, experiment name
    experiment_config: dict, fault configuation of the create fault request.

  Returns:
    FaultinjectiontestingProjectsLocationsExperimentsPatchRequest
  Raises:
    InvalidExperimentConfigurationError: If the experiment config is invalid.
  """
    messages = GetMessagesModule(release_track=base.ReleaseTrack.ALPHA)
    request = messages.FaultinjectiontestingProjectsLocationsExperimentsPatchRequest
    try:
        import_request_message = encoding.DictToMessage({'experiment': experiment_config, 'name': experiment}, request)
    except AttributeError:
        raise InvalidExperimentConfigurationError('An error occurred while parsing the serialized experiment. Please check your input file.')
    unrecognized_field_paths = _GetUnrecognizedFieldPaths(import_request_message)
    if unrecognized_field_paths:
        error_msg_lines = ['Invalid experiment config, the following fields are ' + 'unrecognized:']
        error_msg_lines += unrecognized_field_paths
        raise InvalidExperimentConfigurationError('\n'.join(error_msg_lines))
    return import_request_message