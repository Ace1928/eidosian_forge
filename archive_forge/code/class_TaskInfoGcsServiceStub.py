import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class TaskInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AddTaskEventData = channel.unary_unary('/ray.rpc.TaskInfoGcsService/AddTaskEventData', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddTaskEventDataRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddTaskEventDataReply.FromString)
        self.GetTaskEvents = channel.unary_unary('/ray.rpc.TaskInfoGcsService/GetTaskEvents', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetTaskEventsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetTaskEventsReply.FromString)