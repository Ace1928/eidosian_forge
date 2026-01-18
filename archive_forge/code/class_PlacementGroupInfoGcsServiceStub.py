import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class PlacementGroupInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreatePlacementGroup = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/CreatePlacementGroup', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreatePlacementGroupRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreatePlacementGroupReply.FromString)
        self.RemovePlacementGroup = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/RemovePlacementGroup', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RemovePlacementGroupRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RemovePlacementGroupReply.FromString)
        self.GetPlacementGroup = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/GetPlacementGroup', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetPlacementGroupRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetPlacementGroupReply.FromString)
        self.GetNamedPlacementGroup = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/GetNamedPlacementGroup', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedPlacementGroupRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedPlacementGroupReply.FromString)
        self.GetAllPlacementGroup = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/GetAllPlacementGroup', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllPlacementGroupRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllPlacementGroupReply.FromString)
        self.WaitPlacementGroupUntilReady = channel.unary_unary('/ray.rpc.PlacementGroupInfoGcsService/WaitPlacementGroupUntilReady', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.WaitPlacementGroupUntilReadyRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.WaitPlacementGroupUntilReadyReply.FromString)