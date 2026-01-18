import grpc
from . import instance_manager_pb2 as src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2
@staticmethod
def GetInstanceManagerState(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.InstanceManagerService/GetInstanceManagerState', src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetInstanceManagerStateRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_experimental_dot_instance__manager__pb2.GetInstanceManagerStateReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)