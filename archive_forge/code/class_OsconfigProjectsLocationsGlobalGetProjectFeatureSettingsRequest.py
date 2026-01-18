from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsGlobalGetProjectFeatureSettingsRequest(_messages.Message):
    """A OsconfigProjectsLocationsGlobalGetProjectFeatureSettingsRequest
  object.

  Fields:
    name: Required. Name of the billing config.
      "projects//locations/global/projectFeatureSettings"
  """
    name = _messages.StringField(1, required=True)