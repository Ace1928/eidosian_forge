from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsGlobalUpdateSettingsRequest(_messages.Message):
    """A ConnectorsProjectsLocationsGlobalUpdateSettingsRequest object.

  Fields:
    name: Output only. Resource name of the Connection. Format:
      projects/{project}/locations/global/settings}
    settings: A Settings resource to be passed as the request body.
    updateMask: Required. The list of fields to update.
  """
    name = _messages.StringField(1, required=True)
    settings = _messages.MessageField('Settings', 2)
    updateMask = _messages.StringField(3)