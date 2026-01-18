from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1KeyAliasReference(_messages.Message):
    """A GoogleCloudApigeeV1KeyAliasReference object.

  Fields:
    aliasId: Alias ID. Must exist in the keystore referred to by the
      reference.
    reference: Reference name in the following format:
      `organizations/{org}/environments/{env}/references/{reference}`
  """
    aliasId = _messages.StringField(1)
    reference = _messages.StringField(2)