import grpc
from . import job_agent_pb2 as src_dot_ray_dot_protobuf_dot_job__agent__pb2
def add_JobAgentServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'InitializeJobEnv': grpc.unary_unary_rpc_method_handler(servicer.InitializeJobEnv, request_deserializer=src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.rpc.JobAgentService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))