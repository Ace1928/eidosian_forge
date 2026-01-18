import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class InternalKVGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InternalKVGet = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVGet', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVGetRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVGetReply.FromString)
        self.InternalKVMultiGet = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVMultiGet', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVMultiGetRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVMultiGetReply.FromString)
        self.InternalKVPut = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVPut', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVPutRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVPutReply.FromString)
        self.InternalKVDel = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVDel', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVDelRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVDelReply.FromString)
        self.InternalKVExists = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVExists', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVExistsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVExistsReply.FromString)
        self.InternalKVKeys = channel.unary_unary('/ray.rpc.InternalKVGcsService/InternalKVKeys', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVKeysRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVKeysReply.FromString)