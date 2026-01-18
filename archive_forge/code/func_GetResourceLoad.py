import grpc
from . import node_manager_pb2 as src_dot_ray_dot_protobuf_dot_node__manager__pb2
@staticmethod
def GetResourceLoad(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeManagerService/GetResourceLoad', src_dot_ray_dot_protobuf_dot_node__manager__pb2.GetResourceLoadRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_node__manager__pb2.GetResourceLoadReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)