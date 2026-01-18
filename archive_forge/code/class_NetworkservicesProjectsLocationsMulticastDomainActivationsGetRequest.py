from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastDomainActivationsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastDomainActivationsGetRequest
  object.

  Fields:
    name: Required. The resource name of the multicast domain activation to
      get. Use the following format:
      `projects/*/locations/*/multicastDomainActivations/*`.
  """
    name = _messages.StringField(1, required=True)