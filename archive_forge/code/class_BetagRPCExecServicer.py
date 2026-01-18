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
class BetagRPCExecServicer(object):
    """
    Should we seperate Exec from Config/Oper?


    """

    def ShowCmdTextOutput(self, request, context):
        """Exec commands"""
        context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)

    def ShowCmdJSONOutput(self, request, context):
        context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)