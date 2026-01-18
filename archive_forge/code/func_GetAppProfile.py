import grpc
from google.bigtable.admin.v2 import bigtable_instance_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2
from google.bigtable.admin.v2 import instance_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2
from google.iam.v1 import iam_policy_pb2 as google_dot_iam_dot_v1_dot_iam__policy__pb2
from google.iam.v1 import policy_pb2 as google_dot_iam_dot_v1_dot_policy__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
def GetAppProfile(self, request, context):
    """Gets information about an app profile.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')