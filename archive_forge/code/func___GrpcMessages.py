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
def __GrpcMessages(self, files):
    """Parses the GRPC scoped configuraiton files into their necessary API Gateway message types.

    Args:
      files: Files to be sent in as managed service configs and GRPC service
      definitions

    Returns:
      List of ApigatewayApiConfigFileMessage, list of
      ApigatewayApiConfigGrpcServiceDefinition messages

    Raises:
      BadFileException: If there is something wrong with the files
    """
    grpc_service_definitions = []
    service_configs = []
    for config_file in files:
        config_contents = endpoints.ReadServiceConfigFile(config_file)
        config_dict = self.__ValidJsonOrYaml(config_file, config_contents)
        if config_dict:
            if config_dict.get('type') == 'google.api.Service':
                service_configs.append(self.__MakeApigatewayApiConfigFileMessage(config_contents, config_file))
            else:
                raise calliope_exceptions.BadFileException('The file {} is not a valid api configuration file. The configuration type is expected to be of "google.api.Service".'.format(config_file))
        elif endpoints.IsProtoDescriptor(config_file):
            grpc_service_definitions.append(self.__MakeApigatewayApiConfigGrpcServiceDefinitionMessage(config_contents, config_file))
        elif endpoints.IsRawProto(config_file):
            raise calliope_exceptions.BadFileException('[{}] cannot be used as it is an uncompiled proto file. However, uncompiled proto files can be included for display purposes when compiled as a source for a passed in proto descriptor.'.format(config_file))
        else:
            raise calliope_exceptions.BadFileException('Could not determine the content type of file [{0}]. Supported extensions are .descriptor .json .pb .yaml and .yml'.format(config_file))
    return (service_configs, grpc_service_definitions)