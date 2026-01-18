import grpc
from . import event_pb2 as src_dot_ray_dot_protobuf_dot_event__pb2
def add_ReportEventServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'ReportEvents': grpc.unary_unary_rpc_method_handler(servicer.ReportEvents, request_deserializer=src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.rpc.ReportEventService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))