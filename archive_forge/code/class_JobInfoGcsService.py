import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class JobInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def AddJob(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/AddJob', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddJobRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddJobReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MarkJobFinished(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/MarkJobFinished', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllJobInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/GetAllJobInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllJobInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllJobInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportJobError(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/ReportJobError', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportJobErrorRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportJobErrorReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNextJobID(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobInfoGcsService/GetNextJobID', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNextJobIDRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNextJobIDReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)