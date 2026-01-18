import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
class UserDefinedServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.__call__ = channel.unary_unary('/ray.serve.UserDefinedService/__call__', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString)
        self.Method1 = channel.unary_unary('/ray.serve.UserDefinedService/Method1', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString)
        self.Method2 = channel.unary_unary('/ray.serve.UserDefinedService/Method2', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage2.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse2.FromString)
        self.Streaming = channel.unary_stream('/ray.serve.UserDefinedService/Streaming', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString)