from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementBinauthzVersion(_messages.Message):
    """The version of binauthz.

  Fields:
    webhookVersion: The version of the binauthz webhook.
  """
    webhookVersion = _messages.StringField(1)