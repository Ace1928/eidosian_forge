from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def SetDatabaseIamPolicy(database_ref, policy):
    """Sets the IAM policy on a database."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    policy = iam_util.ParsePolicyFile(policy, msgs.Policy)
    return databases.SetPolicy(database_ref, policy)