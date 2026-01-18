from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsPrivateConnectionsRoutesGetRequest(_messages.Message):
    """A DatastreamProjectsLocationsPrivateConnectionsRoutesGetRequest object.

  Fields:
    name: Required. The name of the Route resource to get.
  """
    name = _messages.StringField(1, required=True)