import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class NodeInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetClusterId(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/GetClusterId', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetClusterIdRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetClusterIdReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterNode(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/RegisterNode', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterNodeRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterNodeReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DrainNode(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/DrainNode', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.DrainNodeRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.DrainNodeReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllNodeInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/GetAllNodeInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllNodeInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllNodeInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetInternalConfig(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/GetInternalConfig', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetInternalConfigRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetInternalConfigReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckAlive(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeInfoGcsService/CheckAlive', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CheckAliveRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CheckAliveReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)