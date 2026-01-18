import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
def CheckAndMutateRow(self, request, context):
    """Mutates a row atomically based on the output of a predicate Reader filter.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')