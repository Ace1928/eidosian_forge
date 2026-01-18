from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsConnectorsResolveInstanceConfigRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsConnectorsResolveInstanceConfigRequest
  object.

  Fields:
    connector: Required. BeyondCorp Connector name using the form:
      `projects/{project_id}/locations/{location_id}/connectors/{connector}`
  """
    connector = _messages.StringField(1, required=True)