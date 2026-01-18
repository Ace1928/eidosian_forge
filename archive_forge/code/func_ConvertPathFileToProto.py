from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def ConvertPathFileToProto(path_file_path, path_file_format):
    """Construct a DataAttributeBindingPath from a JSON/YAML formatted file.

  Args:
    path_file_path: Path to the JSON or YAML file.
    path_file_format: Format for the file provided.
    If file format will not be provided by default it will be json.

  Returns:
    a protorpc.Message of type GoogleCloudDataplexV1DataAttributeBindingPath
    filled in from the JSON or YAML path file.

  Raises:
    BadFileException if the JSON or YAML file is malformed.
  """
    if path_file_format == 'yaml':
        parsed_path = yaml.load(path_file_path)
    else:
        try:
            parsed_path = json.load(path_file_path)
        except ValueError as e:
            raise exceptions.BadFileException('Error parsing JSON: {0}'.format(six.text_type(e)))
    path_message = dataplex_api.GetMessageModule().GoogleCloudDataplexV1DataAttributeBindingPath
    attribute_binding_path = []
    for path in parsed_path['paths']:
        attribute_binding_path.append(encoding.PyValueToMessage(path_message, path))
    return attribute_binding_path