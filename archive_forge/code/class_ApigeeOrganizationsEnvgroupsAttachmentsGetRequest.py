from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvgroupsAttachmentsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvgroupsAttachmentsGetRequest object.

  Fields:
    name: Required. Name of the environment group attachment in the following
      format:
      `organizations/{org}/envgroups/{envgroup}/attachments/{attachment}`
  """
    name = _messages.StringField(1, required=True)