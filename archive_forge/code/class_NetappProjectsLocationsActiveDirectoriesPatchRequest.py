from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsActiveDirectoriesPatchRequest(_messages.Message):
    """A NetappProjectsLocationsActiveDirectoriesPatchRequest object.

  Fields:
    activeDirectory: A ActiveDirectory resource to be passed as the request
      body.
    name: Identifier. The resource name of the active directory. Format: `proj
      ects/{project_number}/locations/{location_id}/activeDirectories/{active_
      directory_id}`.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Active Directory resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    activeDirectory = _messages.MessageField('ActiveDirectory', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)