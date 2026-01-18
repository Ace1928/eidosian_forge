from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsGetCmekConfigRequest(_messages.Message):
    """A CloudtasksProjectsLocationsGetCmekConfigRequest object.

  Fields:
    name: Required. The config. For example:
      projects/PROJECT_ID/locations/LOCATION_ID/CmekConfig`
  """
    name = _messages.StringField(1, required=True)