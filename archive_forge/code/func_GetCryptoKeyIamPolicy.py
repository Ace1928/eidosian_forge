from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def GetCryptoKeyIamPolicy(crypto_key_ref):
    """Fetch the IAM Policy attached to the named CryptoKey.

  Args:
      crypto_key_ref: A resources.Resource naming the CryptoKey.

  Returns:
      An apitools wrapper for the IAM Policy.
  """
    client = base.GetClientInstance()
    messages = base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysGetIamPolicyRequest(options_requestedPolicyVersion=iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION, resource=crypto_key_ref.RelativeName())
    return client.projects_locations_keyRings_cryptoKeys.GetIamPolicy(req)