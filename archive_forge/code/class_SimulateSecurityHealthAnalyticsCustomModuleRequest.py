from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulateSecurityHealthAnalyticsCustomModuleRequest(_messages.Message):
    """Request message to simulate a CustomConfig against a given test
  resource. Maximum size of the request is 4 MB by default.

  Fields:
    customConfig: Required. The custom configuration that you need to test.
    resource: Required. Resource data to simulate custom module against.
  """
    customConfig = _messages.MessageField('GoogleCloudSecuritycenterV1CustomConfig', 1)
    resource = _messages.MessageField('SimulatedResource', 2)