from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetRequest(_messages.Message):
    """A
  NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetRequest
  object.

  Fields:
    name: Required. Name of the ServiceConnectionPolicy to get.
  """
    name = _messages.StringField(1, required=True)