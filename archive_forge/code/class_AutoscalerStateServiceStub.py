import grpc
from . import autoscaler_pb2 as src_dot_ray_dot_protobuf_dot_autoscaler__pb2
class AutoscalerStateServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetClusterResourceState = channel.unary_unary('/ray.rpc.autoscaler.AutoscalerStateService/GetClusterResourceState', request_serializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterResourceStateRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterResourceStateReply.FromString)
        self.ReportAutoscalingState = channel.unary_unary('/ray.rpc.autoscaler.AutoscalerStateService/ReportAutoscalingState', request_serializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateReply.FromString)
        self.RequestClusterResourceConstraint = channel.unary_unary('/ray.rpc.autoscaler.AutoscalerStateService/RequestClusterResourceConstraint', request_serializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.RequestClusterResourceConstraintRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.RequestClusterResourceConstraintReply.FromString)
        self.GetClusterStatus = channel.unary_unary('/ray.rpc.autoscaler.AutoscalerStateService/GetClusterStatus', request_serializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterStatusRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterStatusReply.FromString)
        self.DrainNode = channel.unary_unary('/ray.rpc.autoscaler.AutoscalerStateService/DrainNode', request_serializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.DrainNodeRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_autoscaler__pb2.DrainNodeReply.FromString)