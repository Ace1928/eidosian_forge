from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesDeleteRequest object.

  Fields:
    name: Required. The name of the Study resource to be deleted. Format:
      `projects/{project}/locations/{location}/studies/{study}`
  """
    name = _messages.StringField(1, required=True)