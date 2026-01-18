from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetrieveImportableDomainsResponse(_messages.Message):
    """Deprecated: For more information, see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations). Response for the `RetrieveImportableDomains` method.

  Fields:
    domains: A list of domains that the calling user manages in Google
      Domains.
    nextPageToken: When present, there are more results to retrieve. Set
      `page_token` to this value on a subsequent call to get the next page of
      results.
  """
    domains = _messages.MessageField('Domain', 1, repeated=True)
    nextPageToken = _messages.StringField(2)