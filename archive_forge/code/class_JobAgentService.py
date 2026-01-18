import grpc
from . import job_agent_pb2 as src_dot_ray_dot_protobuf_dot_job__agent__pb2
class JobAgentService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def InitializeJobEnv(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.JobAgentService/InitializeJobEnv', src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)