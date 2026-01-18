from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsUpdateCmekConfigRequest(_messages.Message):
    """A CloudtasksProjectsLocationsUpdateCmekConfigRequest object.

  Fields:
    cmekConfig: A CmekConfig resource to be passed as the request body.
    name: Output only. The config resource name which includes the project and
      location and must end in 'cmekConfig', in the format
      projects/PROJECT_ID/locations/LOCATION_ID/cmekConfig`
    updateMask: List of fields to be updated in this request.
  """
    cmekConfig = _messages.MessageField('CmekConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)