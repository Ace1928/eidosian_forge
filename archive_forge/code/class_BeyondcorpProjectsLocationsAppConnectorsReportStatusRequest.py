from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectorsReportStatusRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectorsReportStatusRequest object.

  Fields:
    appConnector: Required. BeyondCorp Connector name using the form:
      `projects/{project_id}/locations/{location_id}/connectors/{connector}`
    googleCloudBeyondcorpAppconnectorsV1alphaReportStatusRequest: A
      GoogleCloudBeyondcorpAppconnectorsV1alphaReportStatusRequest resource to
      be passed as the request body.
  """
    appConnector = _messages.StringField(1, required=True)
    googleCloudBeyondcorpAppconnectorsV1alphaReportStatusRequest = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1alphaReportStatusRequest', 2)