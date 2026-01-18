from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsGlobalUpdateProjectFeatureSettingsRequest(_messages.Message):
    """A OsconfigProjectsLocationsGlobalUpdateProjectFeatureSettingsRequest
  object.

  Fields:
    name: Required. Immutable. Name of the config, e.g.
      projects/12345/locations/global/projectFeatureSettings
    projectFeatureSettings: A ProjectFeatureSettings resource to be passed as
      the request body.
    updateMask: Optional. Field mask that controls which fields of the
      ProjectFeatureSettings should be updated.
  """
    name = _messages.StringField(1, required=True)
    projectFeatureSettings = _messages.MessageField('ProjectFeatureSettings', 2)
    updateMask = _messages.StringField(3)