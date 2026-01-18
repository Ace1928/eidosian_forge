import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
def MutateRows(self, request, context):
    """Mutates multiple rows in a batch. Each individual row is mutated
    atomically as in MutateRow, but the entire batch is not executed
    atomically.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')