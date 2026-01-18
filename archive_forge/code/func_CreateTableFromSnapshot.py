import grpc
from google.bigtable.admin.v2 import bigtable_table_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2
from google.bigtable.admin.v2 import table_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
def CreateTableFromSnapshot(self, request, context):
    """Creates a new table from the specified snapshot. The target table must
    not exist. The snapshot and the table must be in the same instance.

    Note: This is a private alpha release of Cloud Bigtable snapshots. This
    feature is not currently available to most Cloud Bigtable customers. This
    feature might be changed in backward-incompatible ways and is not
    recommended for production use. It is not subject to any SLA or deprecation
    policy.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')