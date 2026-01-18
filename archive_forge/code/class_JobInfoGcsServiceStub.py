import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class JobInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AddJob = channel.unary_unary('/ray.rpc.JobInfoGcsService/AddJob', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddJobRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddJobReply.FromString)
        self.MarkJobFinished = channel.unary_unary('/ray.rpc.JobInfoGcsService/MarkJobFinished', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.MarkJobFinishedReply.FromString)
        self.GetAllJobInfo = channel.unary_unary('/ray.rpc.JobInfoGcsService/GetAllJobInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllJobInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllJobInfoReply.FromString)
        self.ReportJobError = channel.unary_unary('/ray.rpc.JobInfoGcsService/ReportJobError', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportJobErrorRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportJobErrorReply.FromString)
        self.GetNextJobID = channel.unary_unary('/ray.rpc.JobInfoGcsService/GetNextJobID', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNextJobIDRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNextJobIDReply.FromString)