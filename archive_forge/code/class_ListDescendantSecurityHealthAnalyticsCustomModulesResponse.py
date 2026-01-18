from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDescendantSecurityHealthAnalyticsCustomModulesResponse(_messages.Message):
    """Response message for listing descendant Security Health Analytics custom
  modules.

  Fields:
    nextPageToken: If not empty, indicates that there may be more custom
      modules to be returned.
    securityHealthAnalyticsCustomModules: Custom modules belonging to the
      requested parent and its descendants.
  """
    nextPageToken = _messages.StringField(1)
    securityHealthAnalyticsCustomModules = _messages.MessageField('GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', 2, repeated=True)