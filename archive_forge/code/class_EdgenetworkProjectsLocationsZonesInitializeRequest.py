from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgenetworkProjectsLocationsZonesInitializeRequest(_messages.Message):
    """A EdgenetworkProjectsLocationsZonesInitializeRequest object.

  Fields:
    initializeZoneRequest: A InitializeZoneRequest resource to be passed as
      the request body.
    name: Required. The name of the zone resource.
  """
    initializeZoneRequest = _messages.MessageField('InitializeZoneRequest', 1)
    name = _messages.StringField(2, required=True)