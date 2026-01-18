import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
@staticmethod
def Streaming(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_stream(request, target, '/ray.serve.UserDefinedService/Streaming', src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)