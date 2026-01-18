import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
def _abortion(rpc_error_call):
    code = rpc_error_call.code()
    pair = _STATUS_CODE_TO_ABORTION_KIND_AND_ABORTION_ERROR_CLASS.get(code)
    error_kind = face.Abortion.Kind.LOCAL_FAILURE if pair is None else pair[0]
    return face.Abortion(error_kind, rpc_error_call.initial_metadata(), rpc_error_call.trailing_metadata(), code, rpc_error_call.details())