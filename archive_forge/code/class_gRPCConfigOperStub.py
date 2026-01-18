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
class gRPCConfigOperStub(object):

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.GetConfig = channel.unary_stream('/IOSXRExtensibleManagabilityService.gRPCConfigOper/GetConfig', request_serializer=ConfigGetArgs.SerializeToString, response_deserializer=ConfigGetReply.FromString)
        self.MergeConfig = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/MergeConfig', request_serializer=ConfigArgs.SerializeToString, response_deserializer=ConfigReply.FromString)
        self.DeleteConfig = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/DeleteConfig', request_serializer=ConfigArgs.SerializeToString, response_deserializer=ConfigReply.FromString)
        self.ReplaceConfig = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/ReplaceConfig', request_serializer=ConfigArgs.SerializeToString, response_deserializer=ConfigReply.FromString)
        self.CliConfig = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/CliConfig', request_serializer=CliConfigArgs.SerializeToString, response_deserializer=CliConfigReply.FromString)
        self.CommitReplace = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/CommitReplace', request_serializer=CommitReplaceArgs.SerializeToString, response_deserializer=CommitReplaceReply.FromString)
        self.CommitConfig = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/CommitConfig', request_serializer=CommitArgs.SerializeToString, response_deserializer=CommitReply.FromString)
        self.ConfigDiscardChanges = channel.unary_unary('/IOSXRExtensibleManagabilityService.gRPCConfigOper/ConfigDiscardChanges', request_serializer=DiscardChangesArgs.SerializeToString, response_deserializer=DiscardChangesReply.FromString)
        self.GetOper = channel.unary_stream('/IOSXRExtensibleManagabilityService.gRPCConfigOper/GetOper', request_serializer=GetOperArgs.SerializeToString, response_deserializer=GetOperReply.FromString)
        self.CreateSubs = channel.unary_stream('/IOSXRExtensibleManagabilityService.gRPCConfigOper/CreateSubs', request_serializer=CreateSubsArgs.SerializeToString, response_deserializer=CreateSubsReply.FromString)