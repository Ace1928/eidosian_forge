from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEffectiveSecurityHealthAnalyticsCustomModulesResponse(_messages.Message):
    """Response message for listing effective Security Health Analytics custom
  modules.

  Fields:
    effectiveSecurityHealthAnalyticsCustomModules: Effective custom modules
      belonging to the requested parent.
    nextPageToken: If not empty, indicates that there may be more effective
      custom modules to be returned.
  """
    effectiveSecurityHealthAnalyticsCustomModules = _messages.MessageField('GoogleCloudSecuritycenterV1EffectiveSecurityHealthAnalyticsCustomModule', 1, repeated=True)
    nextPageToken = _messages.StringField(2)