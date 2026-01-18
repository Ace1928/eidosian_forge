from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of studies to return per "page" of
      results. If unspecified, service will pick an appropriate default.
    pageToken: Optional. A page token to request the next page of results. If
      unspecified, there are no subsequent pages.
    parent: Required. The resource name of the Location to list the Study
      from. Format: `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)