import grpc
from . import event_pb2 as src_dot_ray_dot_protobuf_dot_event__pb2
class ReportEventService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ReportEvents(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReportEventService/ReportEvents', src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)