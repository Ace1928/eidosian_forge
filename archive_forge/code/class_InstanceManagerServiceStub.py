import grpc
from . import instance_manager_pb2 as src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2
class InstanceManagerServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetInstanceManagerState = channel.unary_unary('/ray.rpc.autoscaler.InstanceManagerService/GetInstanceManagerState', request_serializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetInstanceManagerStateRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetInstanceManagerStateReply.FromString)
        self.UpdateInstanceManagerState = channel.unary_unary('/ray.rpc.autoscaler.InstanceManagerService/UpdateInstanceManagerState', request_serializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.UpdateInstanceManagerStateRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.UpdateInstanceManagerStateReply.FromString)
        self.GetAvailableInstanceTypes = channel.unary_unary('/ray.rpc.autoscaler.InstanceManagerService/GetAvailableInstanceTypes', request_serializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetAvailableInstanceTypesRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetAvailableInstanceTypesResponse.FromString)