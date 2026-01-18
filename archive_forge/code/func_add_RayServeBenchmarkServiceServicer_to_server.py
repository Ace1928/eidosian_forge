import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
def add_RayServeBenchmarkServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'grpc_call': grpc.unary_unary_rpc_method_handler(servicer.grpc_call, request_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.RawData.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ModelOutput.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.serve.RayServeBenchmarkService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))