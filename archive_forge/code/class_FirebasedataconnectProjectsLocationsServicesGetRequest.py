from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesGetRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesGetRequest object.

  Fields:
    name: Required. The name of the service to retrieve, in the format: ```
      projects/{project}/locations/{location}/services/{service} ```
  """
    name = _messages.StringField(1, required=True)