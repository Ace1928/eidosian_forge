from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalPeeringsDeleteRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalPeeringsDeleteRequest object.

  Fields:
    name: Required. Peering resource name using the form:
      `projects/{project_id}/locations/global/peerings/{peering_id}`
  """
    name = _messages.StringField(1, required=True)