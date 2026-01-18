from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNasJobsNasTrialDetailsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsNasJobsNasTrialDetailsListRequest object.

  Fields:
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      ListNasTrialDetailsResponse.next_page_token of the previous
      JobService.ListNasTrialDetails call.
    parent: Required. The name of the NasJob resource. Format:
      `projects/{project}/locations/{location}/nasJobs/{nas_job}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)