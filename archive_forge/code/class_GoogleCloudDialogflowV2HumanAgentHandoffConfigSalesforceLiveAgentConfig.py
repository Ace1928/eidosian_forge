from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentHandoffConfigSalesforceLiveAgentConfig(_messages.Message):
    """Configuration specific to Salesforce Live Agent.

  Fields:
    buttonId: Required. Live Agent chat button ID.
    deploymentId: Required. Live Agent deployment ID.
    endpointDomain: Required. Domain of the Live Agent endpoint for this
      agent. You can find the endpoint URL in the `Live Agent settings` page.
      For example if URL has the form
      https://d.la4-c2-phx.salesforceliveagent.com/..., you should fill in
      d.la4-c2-phx.salesforceliveagent.com.
    organizationId: Required. The organization ID of the Salesforce account.
  """
    buttonId = _messages.StringField(1)
    deploymentId = _messages.StringField(2)
    endpointDomain = _messages.StringField(3)
    organizationId = _messages.StringField(4)