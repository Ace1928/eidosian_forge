from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def AddInstanceIamPolicyBinding(instance_ref, member, role):
    """Adds a policy binding to an instance IAM policy."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    policy = instances.GetIamPolicy(instance_ref)
    iam_util.AddBindingToIamPolicy(msgs.Binding, policy, member, role)
    return instances.SetPolicy(instance_ref, policy)