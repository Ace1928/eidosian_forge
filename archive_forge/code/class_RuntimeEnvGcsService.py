import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class RuntimeEnvGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def PinRuntimeEnvURI(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RuntimeEnvGcsService/PinRuntimeEnvURI', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)