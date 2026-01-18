from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
def UpdateSshPublicKey(self, user, fingerprint, public_key, update_mask, expiration_time=None):
    """Update an existing SSH public key in a user's login profile.

    Args:
      user: str, The email address of the OS Login user.
      fingerprint: str, The fingerprint of the SSH key to update.
      public_key: str, An SSH public key.
      update_mask: str, A mask to control which fields get updated.
      expiration_time: int, microseconds since epoch.

    Returns:
      The login profile for the user.
    """
    public_key_message = self.messages.SshPublicKey(key=public_key, expirationTimeUsec=expiration_time)
    message = self.messages.OsloginUsersSshPublicKeysPatchRequest(name='users/{0}/sshPublicKeys/{1}'.format(user, fingerprint), sshPublicKey=public_key_message, updateMask=update_mask)
    res = self.client.users_sshPublicKeys.Patch(message)
    return res