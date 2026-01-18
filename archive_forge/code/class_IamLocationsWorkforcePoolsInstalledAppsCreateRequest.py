from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsCreateRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsCreateRequest object.

  Fields:
    parent: Required. The pool to create this workforce pool installed app in.
      Format: `locations/{location}/workforcePools/{workforce_pool}`
    workforcePoolInstalledApp: A WorkforcePoolInstalledApp resource to be
      passed as the request body.
    workforcePoolInstalledAppId: Required. The ID to use for the workforce
      pool installed app, which becomes the final component of the resource
      name. This value should be 4-32 characters, and may contain the
      characters [a-z0-9-]. The prefix `gcp-` is reserved for use by Google,
      and may not be specified.
  """
    parent = _messages.StringField(1, required=True)
    workforcePoolInstalledApp = _messages.MessageField('WorkforcePoolInstalledApp', 2)
    workforcePoolInstalledAppId = _messages.StringField(3)