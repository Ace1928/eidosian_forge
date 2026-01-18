from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesConnectorsGetRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesConnectorsGetRequest
  object.

  Fields:
    name: Required. The name of the connector to retrieve, in the format: ```
      projects/{project}/locations/{location}/services/{service}/connectors/{c
      onnector} ```
  """
    name = _messages.StringField(1, required=True)