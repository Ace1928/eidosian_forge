import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
class RayServeBenchmarkService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def grpc_call(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.serve.RayServeBenchmarkService/grpc_call', src_dot_ray_dot_protobuf_dot_serve__pb2.RawData.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.ModelOutput.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)