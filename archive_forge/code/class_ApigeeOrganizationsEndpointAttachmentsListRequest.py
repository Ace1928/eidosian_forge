from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEndpointAttachmentsListRequest(_messages.Message):
    """A ApigeeOrganizationsEndpointAttachmentsListRequest object.

  Fields:
    pageSize: Optional. Maximum number of endpoint attachments to return. If
      unspecified, at most 25 attachments will be returned.
    pageToken: Optional. Page token, returned from a previous
      `ListEndpointAttachments` call, that you can use to retrieve the next
      page.
    parent: Required. Name of the organization for which to list endpoint
      attachments. Use the following structure in your request:
      `organizations/{org}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)