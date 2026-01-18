from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesSearchRangeRequest(_messages.Message):
    """A ServicenetworkingServicesSearchRangeRequest object.

  Fields:
    parent: Required. This is in a form services/{service}. {service} the name
      of the private access management service, for example 'service-
      peering.example.com'.
    searchRangeRequest: A SearchRangeRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    searchRangeRequest = _messages.MessageField('SearchRangeRequest', 2)