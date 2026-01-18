from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def RemoveInstanceIamPolicyBinding(instance_ref, member, role):
    """Removes a policy binding from an instance IAM policy."""
    policy = instances.GetIamPolicy(instance_ref)
    iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    return instances.SetPolicy(instance_ref, policy)