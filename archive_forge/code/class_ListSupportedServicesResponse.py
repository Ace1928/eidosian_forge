from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSupportedServicesResponse(_messages.Message):
    """A response to `ListSupportedServicesRequest`.

  Fields:
    nextPageToken: The pagination token to retrieve the next page of results.
      If the value is empty, no further results remain.
    supportedServices: List of services supported by VPC-SC instances.
  """
    nextPageToken = _messages.StringField(1)
    supportedServices = _messages.MessageField('SupportedService', 2, repeated=True)