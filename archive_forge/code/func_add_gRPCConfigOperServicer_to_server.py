from __future__ import absolute_import, division, print_function
import sys
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper
import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities
def add_gRPCConfigOperServicer_to_server(servicer, server):
    rpc_method_handlers = {'GetConfig': grpc.unary_stream_rpc_method_handler(servicer.GetConfig, request_deserializer=ConfigGetArgs.FromString, response_serializer=ConfigGetReply.SerializeToString), 'MergeConfig': grpc.unary_unary_rpc_method_handler(servicer.MergeConfig, request_deserializer=ConfigArgs.FromString, response_serializer=ConfigReply.SerializeToString), 'DeleteConfig': grpc.unary_unary_rpc_method_handler(servicer.DeleteConfig, request_deserializer=ConfigArgs.FromString, response_serializer=ConfigReply.SerializeToString), 'ReplaceConfig': grpc.unary_unary_rpc_method_handler(servicer.ReplaceConfig, request_deserializer=ConfigArgs.FromString, response_serializer=ConfigReply.SerializeToString), 'CliConfig': grpc.unary_unary_rpc_method_handler(servicer.CliConfig, request_deserializer=CliConfigArgs.FromString, response_serializer=CliConfigReply.SerializeToString), 'CommitReplace': grpc.unary_unary_rpc_method_handler(servicer.CommitReplace, request_deserializer=CommitReplaceArgs.FromString, response_serializer=CommitReplaceReply.SerializeToString), 'CommitConfig': grpc.unary_unary_rpc_method_handler(servicer.CommitConfig, request_deserializer=CommitArgs.FromString, response_serializer=CommitReply.SerializeToString), 'ConfigDiscardChanges': grpc.unary_unary_rpc_method_handler(servicer.ConfigDiscardChanges, request_deserializer=DiscardChangesArgs.FromString, response_serializer=DiscardChangesReply.SerializeToString), 'GetOper': grpc.unary_stream_rpc_method_handler(servicer.GetOper, request_deserializer=GetOperArgs.FromString, response_serializer=GetOperReply.SerializeToString), 'CreateSubs': grpc.unary_stream_rpc_method_handler(servicer.CreateSubs, request_deserializer=CreateSubsArgs.FromString, response_serializer=CreateSubsReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('IOSXRExtensibleManagabilityService.gRPCConfigOper', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))