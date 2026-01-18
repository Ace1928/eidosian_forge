from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsListRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsListRequest object.

  Fields:
    pageSize: Optional. Maximum number entries to return. If unspecified, at
      most 1000 entries will be returned.
    pageToken: Optional. Page token. If provides, must be a valid AppGroup app
      returned from a previous call that can be used to retrieve the next
      page.
    parent: Required. Name of the AppGroup. Use the following structure in
      your request: `organizations/{org}/appgroups/{app_group_name}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)