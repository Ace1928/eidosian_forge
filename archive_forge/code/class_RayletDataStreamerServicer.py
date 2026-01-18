import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletDataStreamerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Datapath(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')