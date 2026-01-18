import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
def add_RuntimeEnvGcsServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'PinRuntimeEnvURI': grpc.unary_unary_rpc_method_handler(servicer.PinRuntimeEnvURI, request_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.rpc.RuntimeEnvGcsService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))