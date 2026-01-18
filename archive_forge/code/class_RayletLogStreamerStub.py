import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletLogStreamerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Logstream = channel.stream_stream('/ray.rpc.RayletLogStreamer/Logstream', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.LogSettingsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.LogData.FromString)