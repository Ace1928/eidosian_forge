from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsGlobalHubsGroupsGetRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsGlobalHubsGroupsGetRequest object.

  Fields:
    name: Required. The name of the route table resource.
  """
    name = _messages.StringField(1, required=True)