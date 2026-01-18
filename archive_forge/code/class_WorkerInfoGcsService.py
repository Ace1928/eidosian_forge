import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class WorkerInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ReportWorkerFailure(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/ReportWorkerFailure', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportWorkerFailureRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportWorkerFailureReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetWorkerInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/GetWorkerInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetWorkerInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetWorkerInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllWorkerInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/GetAllWorkerInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllWorkerInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllWorkerInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AddWorkerInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/AddWorkerInfo', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddWorkerInfoRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddWorkerInfoReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateWorkerDebuggerPort(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/UpdateWorkerDebuggerPort', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerDebuggerPortRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerDebuggerPortReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateWorkerNumPausedThreads(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.WorkerInfoGcsService/UpdateWorkerNumPausedThreads', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerNumPausedThreadsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerNumPausedThreadsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)