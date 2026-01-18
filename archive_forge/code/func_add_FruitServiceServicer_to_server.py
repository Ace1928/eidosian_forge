import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
def add_FruitServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'FruitStand': grpc.unary_unary_rpc_method_handler(servicer.FruitStand, request_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.FruitAmounts.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.FruitCosts.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.serve.FruitService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))