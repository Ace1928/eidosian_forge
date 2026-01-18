import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
def _blocking_stream_unary(channel, group, method, timeout, with_call, protocol_options, metadata, metadata_transformer, request_iterator, request_serializer, response_deserializer):
    try:
        multi_callable = channel.stream_unary(_common.fully_qualified_method(group, method), request_serializer=request_serializer, response_deserializer=response_deserializer)
        effective_metadata = _effective_metadata(metadata, metadata_transformer)
        if with_call:
            response, call = multi_callable.with_call(request_iterator, timeout=timeout, metadata=_metadata.unbeta(effective_metadata), credentials=_credentials(protocol_options))
            return (response, _Rendezvous(None, None, call))
        else:
            return multi_callable(request_iterator, timeout=timeout, metadata=_metadata.unbeta(effective_metadata), credentials=_credentials(protocol_options))
    except grpc.RpcError as rpc_error_call:
        raise _abortion_error(rpc_error_call)