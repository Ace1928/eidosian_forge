import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
class BigtableStub(object):
    """Service for reading from and writing to existing Bigtable tables.
  """

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.ReadRows = channel.unary_stream('/google.bigtable.v2.Bigtable/ReadRows', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.ReadRowsRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.ReadRowsResponse.FromString)
        self.SampleRowKeys = channel.unary_stream('/google.bigtable.v2.Bigtable/SampleRowKeys', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.SampleRowKeysRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.SampleRowKeysResponse.FromString)
        self.MutateRow = channel.unary_unary('/google.bigtable.v2.Bigtable/MutateRow', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.MutateRowRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.MutateRowResponse.FromString)
        self.MutateRows = channel.unary_stream('/google.bigtable.v2.Bigtable/MutateRows', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.MutateRowsRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.MutateRowsResponse.FromString)
        self.CheckAndMutateRow = channel.unary_unary('/google.bigtable.v2.Bigtable/CheckAndMutateRow', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.CheckAndMutateRowRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.CheckAndMutateRowResponse.FromString)
        self.ReadModifyWriteRow = channel.unary_unary('/google.bigtable.v2.Bigtable/ReadModifyWriteRow', request_serializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.ReadModifyWriteRowRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_v2_dot_bigtable__pb2.ReadModifyWriteRowResponse.FromString)