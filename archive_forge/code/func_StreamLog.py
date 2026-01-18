import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
@staticmethod
def StreamLog(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_stream(request, target, '/ray.rpc.LogService/StreamLog', src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)