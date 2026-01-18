from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def AddPolicyBindingsToCryptoKey(crypto_key_ref, member_roles):
    """Add IAM policy bindings on the CryptoKey.

  Does an atomic Read-Modify-Write, adding the members to the roles. Only calls
  SetIamPolicy if the policy would be different.

  Args:
    crypto_key_ref: A resources.Resource naming the CryptoKey.
    member_roles: List of 2-tuples in the form [(member, role), ...].

  Returns:
    The new IAM Policy.
  """
    messages = base.GetMessagesModule()
    policy = GetCryptoKeyIamPolicy(crypto_key_ref)
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    policy_was_updated = False
    for member, role in member_roles:
        if iam_util.AddBindingToIamPolicy(messages.Binding, policy, member, role):
            policy_was_updated = True
    if policy_was_updated:
        return SetCryptoKeyIamPolicy(crypto_key_ref, policy, update_mask='bindings,etag')
    return policy