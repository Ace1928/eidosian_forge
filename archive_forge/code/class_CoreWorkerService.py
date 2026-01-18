import grpc
from . import core_worker_pb2 as src_dot_ray_dot_protobuf_dot_core__worker__pb2
from . import pubsub_pb2 as src_dot_ray_dot_protobuf_dot_pubsub__pb2
class CoreWorkerService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RayletNotifyGCSRestart(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/RayletNotifyGCSRestart', src_dot_ray_dot_protobuf_dot_core__worker__pb2.RayletNotifyGCSRestartRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.RayletNotifyGCSRestartReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PushTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/PushTask', src_dot_ray_dot_protobuf_dot_core__worker__pb2.PushTaskRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.PushTaskReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DirectActorCallArgWaitComplete(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/DirectActorCallArgWaitComplete', src_dot_ray_dot_protobuf_dot_core__worker__pb2.DirectActorCallArgWaitCompleteRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.DirectActorCallArgWaitCompleteReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetObjectStatus(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/GetObjectStatus', src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetObjectStatusRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetObjectStatusReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WaitForActorOutOfScope(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/WaitForActorOutOfScope', src_dot_ray_dot_protobuf_dot_core__worker__pb2.WaitForActorOutOfScopeRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.WaitForActorOutOfScopeReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PubsubLongPolling(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/PubsubLongPolling', src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubLongPollingRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubLongPollingReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportGeneratorItemReturns(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/ReportGeneratorItemReturns', src_dot_ray_dot_protobuf_dot_core__worker__pb2.ReportGeneratorItemReturnsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.ReportGeneratorItemReturnsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PubsubCommandBatch(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/PubsubCommandBatch', src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubCommandBatchRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubCommandBatchReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateObjectLocationBatch(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/UpdateObjectLocationBatch', src_dot_ray_dot_protobuf_dot_core__worker__pb2.UpdateObjectLocationBatchRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.UpdateObjectLocationBatchReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetObjectLocationsOwner(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/GetObjectLocationsOwner', src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetObjectLocationsOwnerRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetObjectLocationsOwnerReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KillActor(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/KillActor', src_dot_ray_dot_protobuf_dot_core__worker__pb2.KillActorRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.KillActorReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CancelTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/CancelTask', src_dot_ray_dot_protobuf_dot_core__worker__pb2.CancelTaskRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.CancelTaskReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemoteCancelTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/RemoteCancelTask', src_dot_ray_dot_protobuf_dot_core__worker__pb2.RemoteCancelTaskRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.RemoteCancelTaskReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetCoreWorkerStats(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/GetCoreWorkerStats', src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetCoreWorkerStatsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.GetCoreWorkerStatsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def LocalGC(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/LocalGC', src_dot_ray_dot_protobuf_dot_core__worker__pb2.LocalGCRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.LocalGCReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteObjects(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/DeleteObjects', src_dot_ray_dot_protobuf_dot_core__worker__pb2.DeleteObjectsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.DeleteObjectsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SpillObjects(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/SpillObjects', src_dot_ray_dot_protobuf_dot_core__worker__pb2.SpillObjectsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.SpillObjectsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RestoreSpilledObjects(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/RestoreSpilledObjects', src_dot_ray_dot_protobuf_dot_core__worker__pb2.RestoreSpilledObjectsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.RestoreSpilledObjectsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteSpilledObjects(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/DeleteSpilledObjects', src_dot_ray_dot_protobuf_dot_core__worker__pb2.DeleteSpilledObjectsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.DeleteSpilledObjectsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PlasmaObjectReady(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/PlasmaObjectReady', src_dot_ray_dot_protobuf_dot_core__worker__pb2.PlasmaObjectReadyRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.PlasmaObjectReadyReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Exit(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/Exit', src_dot_ray_dot_protobuf_dot_core__worker__pb2.ExitRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.ExitReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AssignObjectOwner(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/AssignObjectOwner', src_dot_ray_dot_protobuf_dot_core__worker__pb2.AssignObjectOwnerRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.AssignObjectOwnerReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NumPendingTasks(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/NumPendingTasks', src_dot_ray_dot_protobuf_dot_core__worker__pb2.NumPendingTasksRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.NumPendingTasksReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)