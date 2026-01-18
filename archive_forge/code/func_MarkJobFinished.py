import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
@staticmethod
def MarkJobFinished(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/MarkJobFinished', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)