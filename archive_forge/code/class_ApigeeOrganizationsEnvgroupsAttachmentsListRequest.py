from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvgroupsAttachmentsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvgroupsAttachmentsListRequest object.

  Fields:
    pageSize: Maximum number of environment group attachments to return. The
      page size defaults to 25.
    pageToken: Page token, returned by a previous
      ListEnvironmentGroupAttachments call, that you can use to retrieve the
      next page.
    parent: Required. Name of the environment group in the following format:
      `organizations/{org}/envgroups/{envgroup}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)