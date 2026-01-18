from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNetworksListNetworkUsageRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNetworksListNetworkUsageRequest
  object.

  Fields:
    location: Required. Parent value (project and location).
  """
    location = _messages.StringField(1, required=True)