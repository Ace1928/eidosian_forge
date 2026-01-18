from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsEndpointAttachmentsListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsEndpointAttachmentsListRequest object.

  Fields:
    filter: Filter.
    orderBy: Order by parameters.
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource od the EndpointAttachment, of the form:
      `projects/*/locations/*`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)