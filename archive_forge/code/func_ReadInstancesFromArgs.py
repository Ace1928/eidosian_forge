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
def ReadInstancesFromArgs(json_request, json_instances, text_instances, limit=None):
    """Reads the instances from the given file path ('-' for stdin).

  Exactly one of json_request, json_instances, text_instances must be given.

  Args:
    json_request: str or None, a path to a file ('-' for stdin) containing
        the JSON body of a prediction request.
    json_instances: str or None, a path to a file ('-' for stdin) containing
        instances in JSON format.
    text_instances: str or None, a path to a file ('-' for stdin) containing
        instances in text format.
    limit: int, the maximum number of instances allowed in the file

  Returns:
    A list of instances.

  Raises:
    InvalidInstancesFileError: If the input file is invalid (invalid format or
        contains too many/zero instances), or an improper combination of input
        files was given.
  """
    mutex_args = [json_request, json_instances, text_instances]
    if len({arg for arg in mutex_args if arg}) != 1:
        raise InvalidInstancesFileError('Exactly one of --json-request, --json-instances and --text-instances must be specified.')
    if json_request:
        data_format = 'json_request'
        input_file = json_request
    if json_instances:
        data_format = 'json'
        input_file = json_instances
    elif text_instances:
        data_format = 'text'
        input_file = text_instances
    data = console_io.ReadFromFileOrStdin(input_file, binary=True)
    with io.BytesIO(data) as f:
        if data_format == 'json_request':
            return ReadRequest(f)
        else:
            return ReadInstances(f, data_format, limit=limit)