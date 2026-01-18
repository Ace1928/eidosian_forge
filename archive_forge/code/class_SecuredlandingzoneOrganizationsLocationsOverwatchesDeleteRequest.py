from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesDeleteRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesDeleteRequest
  object.

  Fields:
    name: Required. The name of the overwatch resource to delete. The format
      is organizations/{org_id}/locations/{location_id}/overwatches/{overwatch
      _id}.
  """
    name = _messages.StringField(1, required=True)