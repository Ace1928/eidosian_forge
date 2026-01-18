from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def RemovePolicyBindingFromEkmConfig(ekm_config_name, member, role):
    """Does an atomic Read-Modify-Write, removing the member from the role."""
    policy = GetEkmConfigIamPolicy(ekm_config_name)
    iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    return SetEkmConfigIamPolicy(ekm_config_name, policy, update_mask='bindings,etag')