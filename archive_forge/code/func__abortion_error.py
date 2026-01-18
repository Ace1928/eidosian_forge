import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
def _abortion_error(rpc_error_call):
    code = rpc_error_call.code()
    pair = _STATUS_CODE_TO_ABORTION_KIND_AND_ABORTION_ERROR_CLASS.get(code)
    exception_class = face.AbortionError if pair is None else pair[1]
    return exception_class(rpc_error_call.initial_metadata(), rpc_error_call.trailing_metadata(), code, rpc_error_call.details())