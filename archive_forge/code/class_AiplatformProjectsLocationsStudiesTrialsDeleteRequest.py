from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsDeleteRequest object.

  Fields:
    name: Required. The Trial's name. Format:
      `projects/{project}/locations/{location}/studies/{study}/trials/{trial}`
  """
    name = _messages.StringField(1, required=True)