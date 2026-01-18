import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
class RayServeAPIServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListApplications = channel.unary_unary('/ray.serve.RayServeAPIService/ListApplications', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ListApplicationsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ListApplicationsResponse.FromString)
        self.Healthz = channel.unary_unary('/ray.serve.RayServeAPIService/Healthz', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.HealthzRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.HealthzResponse.FromString)