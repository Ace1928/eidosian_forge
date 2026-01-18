from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ReferenceConfig(_messages.Message):
    """A GoogleCloudApigeeV1ReferenceConfig object.

  Fields:
    name: Name of the reference in the following format:
      `organizations/{org}/environments/{env}/references/{reference}`
    resourceName: Name of the referenced resource in the following format:
      `organizations/{org}/environments/{env}/keystores/{keystore}` Only
      references to keystore resources are supported.
  """
    name = _messages.StringField(1)
    resourceName = _messages.StringField(2)