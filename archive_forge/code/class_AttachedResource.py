from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachedResource(_messages.Message):
    """Attached resource representation, which is defined by the corresponding
  service provider. It represents an attached resource's payload.

  Fields:
    assetType: The type of this attached resource. Example:
      `osconfig.googleapis.com/Inventory` You can find the supported attached
      asset types of each resource in this table:
      `https://cloud.google.com/asset-inventory/docs/supported-asset-types`
    versionedResources: Versioned resource representations of this attached
      resource. This is repeated because there could be multiple versions of
      the attached resource representations during version migration.
  """
    assetType = _messages.StringField(1)
    versionedResources = _messages.MessageField('VersionedResource', 2, repeated=True)