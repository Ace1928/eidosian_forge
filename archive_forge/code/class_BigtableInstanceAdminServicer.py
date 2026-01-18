import grpc
from google.bigtable.admin.v2 import bigtable_instance_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2
from google.bigtable.admin.v2 import instance_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2
from google.iam.v1 import iam_policy_pb2 as google_dot_iam_dot_v1_dot_iam__policy__pb2
from google.iam.v1 import policy_pb2 as google_dot_iam_dot_v1_dot_policy__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
class BigtableInstanceAdminServicer(object):
    """Service for creating, configuring, and deleting Cloud Bigtable Instances and
  Clusters. Provides access to the Instance and Cluster schemas only, not the
  tables' metadata or data stored in those tables.
  """

    def CreateInstance(self, request, context):
        """Create an instance within a project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetInstance(self, request, context):
        """Gets information about an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListInstances(self, request, context):
        """Lists information about instances in a project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateInstance(self, request, context):
        """Updates an instance within a project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PartialUpdateInstance(self, request, context):
        """Partially updates an instance within a project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteInstance(self, request, context):
        """Delete an instance from a project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateCluster(self, request, context):
        """Creates a cluster within an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetCluster(self, request, context):
        """Gets information about a cluster.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListClusters(self, request, context):
        """Lists information about clusters in an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateCluster(self, request, context):
        """Updates a cluster within an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteCluster(self, request, context):
        """Deletes a cluster from an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateAppProfile(self, request, context):
        """Creates an app profile within an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAppProfile(self, request, context):
        """Gets information about an app profile.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListAppProfiles(self, request, context):
        """Lists information about app profiles in an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateAppProfile(self, request, context):
        """Updates an app profile within an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteAppProfile(self, request, context):
        """Deletes an app profile from an instance.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetIamPolicy(self, request, context):
        """Gets the access control policy for an instance resource. Returns an empty
    policy if an instance exists but does not have a policy set.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetIamPolicy(self, request, context):
        """Sets the access control policy on an instance resource. Replaces any
    existing policy.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestIamPermissions(self, request, context):
        """Returns permissions that the caller has on the specified instance resource.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')