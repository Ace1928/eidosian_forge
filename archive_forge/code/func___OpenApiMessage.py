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
def __OpenApiMessage(self, open_api_specs):
    """Parses the Open API scoped configuraiton files into their necessary API Gateway message types.

    Args:
      open_api_specs: Specs to be used with the API Gateway API Configuration

    Returns:
      List of ApigatewayApiConfigOpenApiDocument messages

    Raises:
      BadFileException: If there is something wrong with the files
    """
    messages = apigateway_base.GetMessagesModule()
    config_files = []
    for config_file in open_api_specs:
        config_contents = endpoints.ReadServiceConfigFile(config_file)
        config_dict = self.__ValidJsonOrYaml(config_file, config_contents)
        if config_dict:
            if 'swagger' in config_dict:
                document = self.__MakeApigatewayApiConfigFileMessage(config_contents, config_file)
                config_files.append(messages.ApigatewayApiConfigOpenApiDocument(document=document))
            elif 'openapi' in config_dict:
                raise calliope_exceptions.BadFileException('API Gateway does not currently support OpenAPI v3 configurations.')
            else:
                raise calliope_exceptions.BadFileException('The file {} is not a valid OpenAPI v2 configuration file.'.format(config_file))
        else:
            raise calliope_exceptions.BadFileException('OpenAPI files should be of JSON or YAML format')
    return config_files