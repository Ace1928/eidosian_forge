from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def RemovePolicyBindingFromKeyRing(key_ring_ref, member, role):
    """Does an atomic Read-Modify-Write, removing the member from the role."""
    policy = GetKeyRingIamPolicy(key_ring_ref)
    iam_util.RemoveBindingFromIamPolicy(policy, member, role)
    return SetKeyRingIamPolicy(key_ring_ref, policy, update_mask='bindings,etag')