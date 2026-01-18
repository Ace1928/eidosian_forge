from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EntityMetadata(_messages.Message):
    """Metadata common to many entities in this API.

  Fields:
    createdAt: Time at which the API proxy was created, in milliseconds since
      epoch.
    lastModifiedAt: Time at which the API proxy was most recently modified, in
      milliseconds since epoch.
    subType: The type of entity described
  """
    createdAt = _messages.IntegerField(1)
    lastModifiedAt = _messages.IntegerField(2)
    subType = _messages.StringField(3)