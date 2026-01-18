import grpc
from . import node_manager_pb2 as src_dot_ray_dot_protobuf_dot_node__manager__pb2
@staticmethod
def GlobalGC(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeManagerService/GlobalGC', src_dot_ray_dot_protobuf_dot_node__manager__pb2.GlobalGCRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_node__manager__pb2.GlobalGCReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)