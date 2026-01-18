from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectorsResolveInstanceConfigRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectorsResolveInstanceConfigRequest
  object.

  Fields:
    appConnector: Required. BeyondCorp AppConnector name using the form: `proj
      ects/{project_id}/locations/{location_id}/appConnectors/{app_connector}`
  """
    appConnector = _messages.StringField(1, required=True)