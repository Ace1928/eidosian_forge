from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsServiceConnectionMapsGetRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsServiceConnectionMapsGetRequest
  object.

  Fields:
    name: Required. Name of the ServiceConnectionMap to get.
  """
    name = _messages.StringField(1, required=True)