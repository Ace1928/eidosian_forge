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
class BetagRPCConfigOperStub(object):

    def GetConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Configuration related commands"""
        raise NotImplementedError()

    def MergeConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    MergeConfig.future = None

    def DeleteConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    DeleteConfig.future = None

    def ReplaceConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    ReplaceConfig.future = None

    def CliConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    CliConfig.future = None

    def CommitReplace(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    CommitReplace.future = None

    def CommitConfig(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Do we need implicit or explicit commit"""
        raise NotImplementedError()
    CommitConfig.future = None

    def ConfigDiscardChanges(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        raise NotImplementedError()
    ConfigDiscardChanges.future = None

    def GetOper(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Get only returns oper data"""
        raise NotImplementedError()

    def CreateSubs(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
        """Get Telemetry Data"""
        raise NotImplementedError()