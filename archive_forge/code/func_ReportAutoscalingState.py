import grpc
from . import autoscaler_pb2 as src_dot_ray_dot_protobuf_dot_autoscaler__pb2
@staticmethod
def ReportAutoscalingState(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/ReportAutoscalingState', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)