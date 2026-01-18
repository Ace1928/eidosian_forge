import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class NodeResourceInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetAllAvailableResources(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeResourceInfoGcsService/GetAllAvailableResources', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllAvailableResourcesRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllAvailableResourcesReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllResourceUsage(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeResourceInfoGcsService/GetAllResourceUsage', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllResourceUsageRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllResourceUsageReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDrainingNodes(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.NodeResourceInfoGcsService/GetDrainingNodes', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetDrainingNodesRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetDrainingNodesReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)