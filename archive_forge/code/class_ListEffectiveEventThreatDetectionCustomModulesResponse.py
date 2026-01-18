from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEffectiveEventThreatDetectionCustomModulesResponse(_messages.Message):
    """Response for listing EffectiveEventThreatDetectionCustomModules.

  Fields:
    effectiveEventThreatDetectionCustomModules: Effective custom modules
      belonging to the requested parent.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    effectiveEventThreatDetectionCustomModules = _messages.MessageField('EffectiveEventThreatDetectionCustomModule', 1, repeated=True)
    nextPageToken = _messages.StringField(2)