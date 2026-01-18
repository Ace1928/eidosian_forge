import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
def add_RayServeAPIServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'ListApplications': grpc.unary_unary_rpc_method_handler(servicer.ListApplications, request_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ListApplicationsRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ListApplicationsResponse.SerializeToString), 'Healthz': grpc.unary_unary_rpc_method_handler(servicer.Healthz, request_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.HealthzRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.HealthzResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.serve.RayServeAPIService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))