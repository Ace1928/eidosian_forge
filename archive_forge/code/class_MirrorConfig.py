from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirrorConfig(_messages.Message):
    """Configuration to automatically mirror a repository from another hosting
  service, for example GitHub or Bitbucket.

  Fields:
    deployKeyId: ID of the SSH deploy key at the other hosting service.
      Removing this key from the other service would deauthorize Google Cloud
      Source Repositories from mirroring.
    url: URL of the main repository at the other hosting service.
    webhookId: ID of the webhook listening to updates to trigger mirroring.
      Removing this webhook from the other hosting service will stop Google
      Cloud Source Repositories from receiving notifications, and thereby
      disabling mirroring.
  """
    deployKeyId = _messages.StringField(1)
    url = _messages.StringField(2)
    webhookId = _messages.StringField(3)