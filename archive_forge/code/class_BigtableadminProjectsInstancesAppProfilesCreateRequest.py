from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesAppProfilesCreateRequest(_messages.Message):
    """A BigtableadminProjectsInstancesAppProfilesCreateRequest object.

  Fields:
    appProfile: A AppProfile resource to be passed as the request body.
    appProfileId: Required. The ID to be used when referring to the new app
      profile within its instance, e.g., just `myprofile` rather than
      `projects/myproject/instances/myinstance/appProfiles/myprofile`.
    ignoreWarnings: If true, ignore safety checks when creating the app
      profile.
    parent: Required. The unique name of the instance in which to create the
      new app profile. Values are of the form
      `projects/{project}/instances/{instance}`.
  """
    appProfile = _messages.MessageField('AppProfile', 1)
    appProfileId = _messages.StringField(2)
    ignoreWarnings = _messages.BooleanField(3)
    parent = _messages.StringField(4, required=True)