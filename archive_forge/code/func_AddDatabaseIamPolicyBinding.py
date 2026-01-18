from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def AddDatabaseIamPolicyBinding(database_ref, member, role):
    """Adds a policy binding to a database IAM policy."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    policy = databases.GetIamPolicy(database_ref)
    iam_util.AddBindingToIamPolicy(msgs.Binding, policy, member, role)
    return databases.SetPolicy(database_ref, policy)