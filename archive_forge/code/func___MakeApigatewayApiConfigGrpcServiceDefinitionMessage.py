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
def __MakeApigatewayApiConfigGrpcServiceDefinitionMessage(self, proto_desc_contents, proto_desc_file):
    """Constructs a GrpcServiceDefinition message from a proto descriptor and the provided list of input files.

    Args:
      proto_desc_contents: The contents of the proto descriptor file.
      proto_desc_file: The path to the proto descriptor file.

    Returns:
      The constructed ApigatewayApiConfigGrpcServiceDefinition message.
    """
    messages = apigateway_base.GetMessagesModule()
    fds = descriptor.FileDescriptorSet.FromString(proto_desc_contents)
    proto_desc_dir = os.path.dirname(proto_desc_file)
    grpc_sources = []
    included_source_paths = []
    not_included_source_paths = []
    for file_descriptor in fds.file:
        source_path = os.path.join(proto_desc_dir, file_descriptor.name)
        if os.path.exists(source_path):
            source_contents = endpoints.ReadServiceConfigFile(source_path)
            file = self.__MakeApigatewayApiConfigFileMessage(source_contents, source_path)
            included_source_paths.append(source_path)
            grpc_sources.append(file)
        else:
            not_included_source_paths.append(source_path)
    if not_included_source_paths:
        log.warning("Proto descriptor's source protos [{0}] were not found on the file system and will not be included in the submitted GRPC service definition. If you meant to include these files, ensure the proto compiler was invoked in the same directory where the proto descriptor [{1}] now resides.".format(', '.join(not_included_source_paths), proto_desc_file))
    if included_source_paths:
        log.info('Added the source protos [{0}] to the GRPC service definition for the provided proto descriptor [{1}].'.format(', '.join(included_source_paths), proto_desc_file))
    file_descriptor_set = self.__MakeApigatewayApiConfigFileMessage(proto_desc_contents, proto_desc_file, True)
    return messages.ApigatewayApiConfigGrpcServiceDefinition(fileDescriptorSet=file_descriptor_set, source=grpc_sources)