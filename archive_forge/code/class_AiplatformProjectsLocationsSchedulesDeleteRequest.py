from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSchedulesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsSchedulesDeleteRequest object.

  Fields:
    name: Required. The name of the Schedule resource to be deleted. Format:
      `projects/{project}/locations/{location}/schedules/{schedule}`
  """
    name = _messages.StringField(1, required=True)