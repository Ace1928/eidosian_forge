from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeployment(_messages.Message):
    """The Deployment.

  Fields:
    displayName: The deployment's display name.
    frns: Output only. The FCC Registration Numbers (FRNs) copied from its
      direct parent.
    name: Output only. Resource name.
    sasUserIds: User ID used by the devices belonging to this deployment. Each
      deployment should be associated with one unique user ID.
  """
    displayName = _messages.StringField(1)
    frns = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    sasUserIds = _messages.StringField(4, repeated=True)