from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOrgPoliciesResponse(_messages.Message):
    """The response returned from the `ListOrgPolicies` method. It will be
  empty if no `Policies` are set on the resource.

  Fields:
    nextPageToken: Page token used to retrieve the next page. This is
      currently not used, but the server may at any point start supplying a
      valid token.
    policies: The `Policies` that are set on the resource. It will be empty if
      no `Policies` are set.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('OrgPolicy', 2, repeated=True)