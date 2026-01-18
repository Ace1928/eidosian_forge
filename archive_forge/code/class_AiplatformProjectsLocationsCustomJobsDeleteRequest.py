from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsCustomJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsCustomJobsDeleteRequest object.

  Fields:
    name: Required. The name of the CustomJob resource to be deleted. Format:
      `projects/{project}/locations/{location}/customJobs/{custom_job}`
  """
    name = _messages.StringField(1, required=True)