from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTlsInspectionPoliciesResponse(_messages.Message):
    """Response returned by the ListTlsInspectionPolicies method.

  Fields:
    nextPageToken: If there might be more results than those appearing in this
      response, then 'next_page_token' is included. To get the next set of
      results, call this method again using the value of 'next_page_token' as
      'page_token'.
    tlsInspectionPolicies: List of TlsInspectionPolicies resources.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    tlsInspectionPolicies = _messages.MessageField('TlsInspectionPolicy', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)