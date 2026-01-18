from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base
from googlecloudsdk.command_lib.iam import iam_util
def TestCryptoKeyIamPermissions(crypto_key_ref, permissions):
    """Return permissions that the caller has on the named CryptoKey."""
    client = base.GetClientInstance()
    messages = base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysTestIamPermissionsRequest(resource=crypto_key_ref.RelativeName(), testIamPermissionsRequest=messages.TestIamPermissionsRequest(permissions=permissions))
    return client.projects_locations_keyRings_cryptoKeys.TestIamPermissions(req)