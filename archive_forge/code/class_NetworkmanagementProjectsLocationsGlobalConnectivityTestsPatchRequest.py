from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsGlobalConnectivityTestsPatchRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsGlobalConnectivityTestsPatchRequest
  object.

  Fields:
    connectivityTest: A ConnectivityTest resource to be passed as the request
      body.
    name: Required. Unique name of the resource using the form:
      `projects/{project_id}/locations/global/connectivityTests/{test_id}`
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
  """
    connectivityTest = _messages.MessageField('ConnectivityTest', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)