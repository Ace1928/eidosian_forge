from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import cloudsdk.google.protobuf.descriptor_pb2 as descriptor
from googlecloudsdk.api_lib.api_gateway import api_configs as api_configs_client
from googlecloudsdk.api_lib.api_gateway import apis as apis_client
from googlecloudsdk.api_lib.api_gateway import base as apigateway_base
from googlecloudsdk.api_lib.api_gateway import operations as operations_client
from googlecloudsdk.api_lib.endpoints import services_util as endpoints
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.api_gateway import common_flags
from googlecloudsdk.command_lib.api_gateway import operations_util
from googlecloudsdk.command_lib.api_gateway import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import http_encoding
def __ValidJsonOrYaml(self, file_name, file_contents):
    """Whether or not this is a valid json or yaml file.

    Args:
      file_name: Name of the file
      file_contents: data for the file

    Returns:
      Boolean for whether or not this is a JSON or YAML

    Raises:
      BadFileException: File appears to be json or yaml but cannot be parsed.
    """
    if endpoints.FilenameMatchesExtension(file_name, ['.json', '.yaml', '.yml']):
        config_dict = endpoints.LoadJsonOrYaml(file_contents)
        if config_dict:
            return config_dict
        else:
            raise calliope_exceptions.BadFileException('Could not read JSON or YAML from config file [{0}].'.format(file_name))
    else:
        return False