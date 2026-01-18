from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def AddPolicyBindingToEkmConfig(ekm_config_name, member, role):
    """Does an atomic Read-Modify-Write, adding the member to the role."""
    messages = base.GetMessagesModule()
    policy = GetEkmConfigIamPolicy(ekm_config_name)
    iam_util.AddBindingToIamPolicy(messages.Binding, policy, member, role)
    return SetEkmConfigIamPolicy(ekm_config_name, policy, update_mask='bindings,etag')