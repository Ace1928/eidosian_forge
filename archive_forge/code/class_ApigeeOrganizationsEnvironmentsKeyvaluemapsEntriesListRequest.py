from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeyvaluemapsEntriesListRequest object.

  Fields:
    pageSize: Optional. Maximum number of key value entries to return. If
      unspecified, at most 100 entries will be returned.
    pageToken: Optional. Page token. If provides, must be a valid key value
      entry returned from a previous call that can be used to retrieve the
      next page.
    parent: Required. Scope as indicated by the URI in which to list key value
      maps. Use **one** of the following structures in your request: *
      `organizations/{organization}/apis/{api}/keyvaluemaps/{keyvaluemap}`. *
      `organizations/{organization}/environments/{environment}/keyvaluemaps/{k
      eyvaluemap}` *
      `organizations/{organization}/keyvaluemaps/{keyvaluemap}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)