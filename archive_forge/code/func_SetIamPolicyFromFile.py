from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_fusion import datafusion as df
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.data_fusion import data_fusion_iam_util
from googlecloudsdk.command_lib.data_fusion import resource_args
from googlecloudsdk.command_lib.iam import iam_util
def SetIamPolicyFromFile(instance_ref, namespace, policy_file, messages, client):
    """Reads an instance's IAM policy from a file, and sets it."""
    new_iam_policy = data_fusion_iam_util.ParsePolicyFile(policy_file, messages.Policy)
    return data_fusion_iam_util.DoSetIamPolicy(instance_ref, namespace, new_iam_policy, messages, client)