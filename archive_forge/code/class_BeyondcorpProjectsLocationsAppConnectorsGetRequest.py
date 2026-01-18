from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectorsGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectorsGetRequest object.

  Fields:
    name: Required. BeyondCorp AppConnector name using the form: `projects/{pr
      oject_id}/locations/{location_id}/appConnectors/{app_connector_id}`
  """
    name = _messages.StringField(1, required=True)