from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsGetRequest object.

  Fields:
    name: Required. The name of the Trial resource. Format:
      `projects/{project}/locations/{location}/studies/{study}/trials/{trial}`
  """
    name = _messages.StringField(1, required=True)