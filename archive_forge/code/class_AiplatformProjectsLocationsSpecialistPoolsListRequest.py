from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSpecialistPoolsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsSpecialistPoolsListRequest object.

  Fields:
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained by
      ListSpecialistPoolsResponse.next_page_token of the previous
      SpecialistPoolService.ListSpecialistPools call. Return first page if
      empty.
    parent: Required. The name of the SpecialistPool's parent resource.
      Format: `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read. FieldMask represents a set
      of
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    readMask = _messages.StringField(4)