import grpc
from google.bigtable.admin.v2 import bigtable_table_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2
from google.bigtable.admin.v2 import table_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
class BigtableTableAdminServicer(object):
    """Service for creating, configuring, and deleting Cloud Bigtable tables.


  Provides access to the table schemas only, not the data stored within
  the tables.
  """

    def CreateTable(self, request, context):
        """Creates a new table in the specified instance.
    The table can be created with a full set of initial column families,
    specified in the request.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

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

    def ListTables(self, request, context):
        """Lists all tables served from a specified instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTable(self, request, context):
        """Gets metadata information about the specified table.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteTable(self, request, context):
        """Permanently deletes a specified table and all of its data.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModifyColumnFamilies(self, request, context):
        """Performs a series of column family modifications on the specified table.
    Either all or none of the modifications will occur before this method
    returns, but data requests received prior to that point may see a table
    where only some modifications have taken effect.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DropRowRange(self, request, context):
        """Permanently drop/delete a row range from a specified table. The request can
    specify whether to delete all rows in a table, or only those that match a
    particular prefix.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GenerateConsistencyToken(self, request, context):
        """Generates a consistency token for a Table, which can be used in
    CheckConsistency to check whether mutations to the table that finished
    before this call started have been replicated. The tokens will be available
    for 90 days.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckConsistency(self, request, context):
        """Checks replication consistency based on a consistency token, that is, if
    replication has caught up based on the conditions specified in the token
    and the check request.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SnapshotTable(self, request, context):
        """Creates a new snapshot in the specified cluster from the specified
    source table. The cluster and the table must be in the same instance.

    Note: This is a private alpha release of Cloud Bigtable snapshots. This
    feature is not currently available to most Cloud Bigtable customers. This
    feature might be changed in backward-incompatible ways and is not
    recommended for production use. It is not subject to any SLA or deprecation
    policy.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSnapshot(self, request, context):
        """Gets metadata information about the specified snapshot.

    Note: This is a private alpha release of Cloud Bigtable snapshots. This
    feature is not currently available to most Cloud Bigtable customers. This
    feature might be changed in backward-incompatible ways and is not
    recommended for production use. It is not subject to any SLA or deprecation
    policy.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListSnapshots(self, request, context):
        """Lists all snapshots associated with the specified cluster.

    Note: This is a private alpha release of Cloud Bigtable snapshots. This
    feature is not currently available to most Cloud Bigtable customers. This
    feature might be changed in backward-incompatible ways and is not
    recommended for production use. It is not subject to any SLA or deprecation
    policy.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteSnapshot(self, request, context):
        """Permanently deletes the specified snapshot.

    Note: This is a private alpha release of Cloud Bigtable snapshots. This
    feature is not currently available to most Cloud Bigtable customers. This
    feature might be changed in backward-incompatible ways and is not
    recommended for production use. It is not subject to any SLA or deprecation
    policy.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')