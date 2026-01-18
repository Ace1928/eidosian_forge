import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
class ReporterServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetProfilingStats = channel.unary_unary('/ray.rpc.ReporterService/GetProfilingStats', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsReply.FromString)
        self.ReportMetrics = channel.unary_unary('/ray.rpc.ReporterService/ReportMetrics', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsReply.FromString)
        self.ReportOCMetrics = channel.unary_unary('/ray.rpc.ReporterService/ReportOCMetrics', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.FromString)
        self.GetTraceback = channel.unary_unary('/ray.rpc.ReporterService/GetTraceback', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackReply.FromString)
        self.CpuProfiling = channel.unary_unary('/ray.rpc.ReporterService/CpuProfiling', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.FromString)