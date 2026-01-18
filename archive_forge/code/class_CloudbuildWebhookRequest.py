from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildWebhookRequest(_messages.Message):
    """A CloudbuildWebhookRequest object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    webhookKey: For GitHub Enterprise webhooks, this key is used to associate
      the webhook request with the GitHubEnterpriseConfig to use for
      validation.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    webhookKey = _messages.StringField(2)