from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsProjectsSettingsLookupEffectiveValueRequest(_messages.Message):
    """A ResourcesettingsProjectsSettingsLookupEffectiveValueRequest object.

  Fields:
    parent: The setting for which an effective value will be evaluated. See
      Setting for naming requirements.
  """
    parent = _messages.StringField(1, required=True)