from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest
  object.

  Fields:
    connectivityTest: A ConnectivityTest resource to be passed as the request
      body.
    parent: Required. The parent resource of the Connectivity Test to create:
      `projects/{project_id}/locations/global`
    testId: Required. The logical name of the Connectivity Test in your
      project with the following restrictions: * Must contain only lowercase
      letters, numbers, and hyphens. * Must start with a letter. * Must be
      between 1-40 characters. * Must end with a number or a letter. * Must be
      unique within the customer project
  """
    connectivityTest = _messages.MessageField('ConnectivityTest', 1)
    parent = _messages.StringField(2, required=True)
    testId = _messages.StringField(3)