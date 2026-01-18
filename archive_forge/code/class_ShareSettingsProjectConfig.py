from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShareSettingsProjectConfig(_messages.Message):
    """Config for each project in the share settings.

  Fields:
    projectId: The project ID, should be same as the key of this project
      config in the parent map.
  """
    projectId = _messages.StringField(1)