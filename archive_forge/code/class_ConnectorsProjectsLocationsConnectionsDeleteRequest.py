from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsDeleteRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsDeleteRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/connections/*`
  """
    name = _messages.StringField(1, required=True)