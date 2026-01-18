from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsUndeleteRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsUndeleteRequest object.

  Fields:
    name: Required. The name of the workforce pool installed app to undelete.
      Format: `locations/{location}/workforcePools/{workforce_pool}/installedA
      pps/{installed_app}`
    undeleteWorkforcePoolInstalledAppRequest: A
      UndeleteWorkforcePoolInstalledAppRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkforcePoolInstalledAppRequest = _messages.MessageField('UndeleteWorkforcePoolInstalledAppRequest', 2)