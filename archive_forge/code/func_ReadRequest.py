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
def ReadRequest(input_file):
    """Reads a JSON request from the specified input file.

  Args:
    input_file: An open file-like object for the input file.

  Returns:
    A list of instances.

  Raises:
    InvalidInstancesFileError: If the input file is invalid.
  """
    contents = input_file.read()
    if isinstance(contents, six.binary_type):
        contents = encoding.Decode(contents, encoding='utf-8-sig')
    try:
        request = json.loads(contents)
    except ValueError:
        raise InvalidInstancesFileError('Input instances are not in JSON format. See "gcloud ml-engine predict --help" for details.')
    if 'instances' not in request:
        raise InvalidInstancesFileError('Invalid JSON request: missing "instances" attribute')
    instances = request['instances']
    if not isinstance(instances, list):
        raise InvalidInstancesFileError('Invalid JSON request: "instances" must be a list')
    return instances