import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletDriver(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Init(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/Init', src_dot_ray_dot_protobuf_dot_ray__client__pb2.InitRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.InitResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PrepRuntimeEnv(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/PrepRuntimeEnv', src_dot_ray_dot_protobuf_dot_ray__client__pb2.PrepRuntimeEnvRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.PrepRuntimeEnvResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetObject(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/ray.rpc.RayletDriver/GetObject', src_dot_ray_dot_protobuf_dot_ray__client__pb2.GetRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.GetResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PutObject(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/PutObject', src_dot_ray_dot_protobuf_dot_ray__client__pb2.PutRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.PutResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WaitObject(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/WaitObject', src_dot_ray_dot_protobuf_dot_ray__client__pb2.WaitRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.WaitResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Schedule(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/Schedule', src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientTask.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientTaskTicket.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Terminate(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/Terminate', src_dot_ray_dot_protobuf_dot_ray__client__pb2.TerminateRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.TerminateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClusterInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/ClusterInfo', src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClusterInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClusterInfoResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KVGet(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/KVGet', src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVGetRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVGetResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KVPut(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/KVPut', src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVPutRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVPutResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KVDel(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/KVDel', src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVDelRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVDelResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KVList(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/KVList', src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVListRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVListResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KVExists(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/KVExists', src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVExistsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVExistsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListNamedActors(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/ListNamedActors', src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientListNamedActorsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientListNamedActorsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PinRuntimeEnvURI(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.RayletDriver/PinRuntimeEnvURI', src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientPinRuntimeEnvURIRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientPinRuntimeEnvURIResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)