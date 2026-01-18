import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class ActorInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RegisterActor(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/RegisterActor', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterActorRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterActorReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateActor(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/CreateActor', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreateActorRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreateActorReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetActorInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/GetActorInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetActorInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetActorInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNamedActorInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/GetNamedActorInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedActorInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedActorInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListNamedActors(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/ListNamedActors', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ListNamedActorsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ListNamedActorsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllActorInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/GetAllActorInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllActorInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllActorInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KillActorViaGcs(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ActorInfoGcsService/KillActorViaGcs', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.KillActorViaGcsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.KillActorViaGcsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)