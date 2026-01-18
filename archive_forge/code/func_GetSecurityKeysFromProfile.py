from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
def GetSecurityKeysFromProfile(user, oslogin_client, profile=None):
    """Return a list of 'private' security keys from the OS Login Profile."""
    if not profile:
        profile = oslogin_client.GetLoginProfile(user)
    sk_list = []
    if not hasattr(profile, 'securityKeys') or not profile.securityKeys:
        return []
    for security_key in profile.securityKeys:
        sk_list.append(security_key.privateKey)
    return sk_list