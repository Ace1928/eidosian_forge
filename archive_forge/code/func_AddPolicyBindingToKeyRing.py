from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def AddPolicyBindingToKeyRing(key_ring_ref, member, role):
    """Does an atomic Read-Modify-Write, adding the member to the role."""
    messages = base.GetMessagesModule()
    policy = GetKeyRingIamPolicy(key_ring_ref)
    iam_util.AddBindingToIamPolicy(messages.Binding, policy, member, role)
    return SetKeyRingIamPolicy(key_ring_ref, policy, update_mask='bindings,etag')