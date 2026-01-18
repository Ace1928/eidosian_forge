from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOrgMembershipsResponse(_messages.Message):
    """The response message for OrgMembershipsService.ListOrgMemberships.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is empty, there are no subsequent pages.
    orgMemberships: The non-vacuous membership in an orgUnit.
  """
    nextPageToken = _messages.StringField(1)
    orgMemberships = _messages.MessageField('OrgMembership', 2, repeated=True)