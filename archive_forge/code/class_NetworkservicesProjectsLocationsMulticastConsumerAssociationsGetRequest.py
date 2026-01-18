from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastConsumerAssociationsGetRequest(_messages.Message):
    """A
  NetworkservicesProjectsLocationsMulticastConsumerAssociationsGetRequest
  object.

  Fields:
    name: Required. The resource name of the multicast consumer association to
      get. Use the following format:
      `projects/*/locations/*/multicastConsumerAssociations/*`.
  """
    name = _messages.StringField(1, required=True)