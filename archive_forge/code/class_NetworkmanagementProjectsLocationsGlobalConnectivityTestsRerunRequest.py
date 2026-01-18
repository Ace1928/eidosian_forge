from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsGlobalConnectivityTestsRerunRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsGlobalConnectivityTestsRerunRequest
  object.

  Fields:
    name: Required. Connectivity Test resource name using the form:
      `projects/{project_id}/locations/global/connectivityTests/{test_id}`
    rerunConnectivityTestRequest: A RerunConnectivityTestRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    rerunConnectivityTestRequest = _messages.MessageField('RerunConnectivityTestRequest', 2)