from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PscAutomatedEndpoints(_messages.Message):
    """PscAutomatedEndpoints defines the output of the forwarding rule
  automatically created by each PscAutomationConfig.

  Fields:
    matchAddress: Ip Address created by the automated forwarding rule.
    network: Corresponding network in pscAutomationConfigs.
    projectId: Corresponding project_id in pscAutomationConfigs
  """
    matchAddress = _messages.StringField(1)
    network = _messages.StringField(2)
    projectId = _messages.StringField(3)