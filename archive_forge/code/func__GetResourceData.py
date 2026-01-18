from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def _GetResourceData(self, data, request_data):
    """Gets the resource data from the arguments and request data.

    This a temporary method to align the old and new schemas and should be
    removed after b/272076207 is complete.

    Args:
      data: arguments yaml data in command.
      request_data: request yaml data in command.

    Returns:
      resource data with missing request params.

    Raises:
      InvalidSchemaError: if the YAML command is malformed.
    """
    request_data = request_data or {}
    resource = data.get('resource')
    if not resource:
        return []
    moved_request_params = ['resource_method_params', 'parse_resource_into_request', 'use_relative_name']
    for request_param in moved_request_params:
        param = request_data.get(request_param)
        if param is not None:
            if resource.get(request_param) is not None:
                raise util.InvalidSchemaError('[{}] is defined in both request and argument.param. Recommend only defining in argument.param'.format(request_param))
            resource[request_param] = param
    resource['resource_spec'] = resource.get('spec', {})
    return [resource]