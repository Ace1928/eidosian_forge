from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def BuildMessageFromKrmData(krm_data, field_mappings, collection, request_method, api_version=None, static_fields=None):
    """Build a Apitools message for specified method from KRM Yaml.

  Args:
      krm_data: {string: string}, A YAML like object containing the
        message data.
      field_mappings: {string: ApitoolsToKrmFieldDescriptor}, A mapping from
        message field names to mapping descriptors.
      collection: The resource collection of the requests method. Together with
        request_method, determine the actual message to generate.
      request_method: The api method whose request message we want to generate.
      api_version: Version of the api to retrieve the message type from. If None
        will use default API version.
      static_fields: {string: object}, Additional fields to set in the
        message that are not mapped from data. Including calculated fields
        and static values.

  Returns:
    The instantiated apitools Message with all fields populated from data.
  """
    method = registry.GetMethod(collection, request_method, api_version)
    request_class = method.GetRequestType()
    return ParseMessageFromDict(krm_data, field_mappings, request_class, additional_fields=static_fields)