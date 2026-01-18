import collections
import sys
import types
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import grpc
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import SerializingFunction
def _with_call(self, request_iterator: RequestIterableType, timeout: Optional[float]=None, metadata: Optional[MetadataType]=None, credentials: Optional[grpc.CallCredentials]=None, wait_for_ready: Optional[bool]=None, compression: Optional[grpc.Compression]=None) -> Tuple[Any, grpc.Call]:
    client_call_details = _ClientCallDetails(self._method, timeout, metadata, credentials, wait_for_ready, compression)

    def continuation(new_details, request_iterator):
        new_method, new_timeout, new_metadata, new_credentials, new_wait_for_ready, new_compression = _unwrap_client_call_details(new_details, client_call_details)
        try:
            response, call = self._thunk(new_method).with_call(request_iterator, timeout=new_timeout, metadata=new_metadata, credentials=new_credentials, wait_for_ready=new_wait_for_ready, compression=new_compression)
            return _UnaryOutcome(response, call)
        except grpc.RpcError as rpc_error:
            return rpc_error
        except Exception as exception:
            return _FailureOutcome(exception, sys.exc_info()[2])
    call = self._interceptor.intercept_stream_unary(continuation, client_call_details, request_iterator)
    return (call.result(), call)