from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkConfig(_messages.Message):
    """A LinkConfig object.

  Fields:
    assetType: Required. Reference to the asset type name of the linked asset,
      in the following form:
      `projects/{project}/locations/{location}/assetTypes/{name}`.
    owner: Output only. The owner of the link, if it's updated by the system.
    required: If true, this asset link is required during asset creation.
  """
    assetType = _messages.StringField(1)
    owner = _messages.StringField(2)
    required = _messages.BooleanField(3)