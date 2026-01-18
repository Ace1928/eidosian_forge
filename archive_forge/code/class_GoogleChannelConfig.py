from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleChannelConfig(_messages.Message):
    """A GoogleChannelConfig is a resource that stores the custom settings
  respected by Eventarc first-party triggers in the matching region. Once
  configured, first-party event data will be protected using the specified
  custom managed encryption key instead of Google-managed encryption keys.

  Fields:
    cryptoKeyName: Optional. Resource name of a KMS crypto key (managed by the
      user) used to encrypt/decrypt their event data. It must match the
      pattern `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
    name: Required. The resource name of the config. Must be in the format of,
      `projects/{project}/locations/{location}/googleChannelConfig`.
    updateTime: Output only. The last-modified time.
  """
    cryptoKeyName = _messages.StringField(1)
    name = _messages.StringField(2)
    updateTime = _messages.StringField(3)