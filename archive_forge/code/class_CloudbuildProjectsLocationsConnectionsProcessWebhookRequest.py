from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsProcessWebhookRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsProcessWebhookRequest object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    parent: Required. Project and location where the webhook will be received.
      Format: `projects/*/locations/*`.
    webhookKey: Arbitrary additional key to find the maching repository for a
      webhook event if needed.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    parent = _messages.StringField(2, required=True)
    webhookKey = _messages.StringField(3)