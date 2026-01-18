import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletDataStreamerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Datapath = channel.stream_stream('/ray.rpc.RayletDataStreamer/Datapath', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.DataRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.DataResponse.FromString)