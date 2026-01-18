from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesAppProfilesDeleteRequest(_messages.Message):
    """A BigtableadminProjectsInstancesAppProfilesDeleteRequest object.

  Fields:
    ignoreWarnings: Required. If true, ignore safety checks when deleting the
      app profile.
    name: Required. The unique name of the app profile to be deleted. Values
      are of the form
      `projects/{project}/instances/{instance}/appProfiles/{app_profile}`.
  """
    ignoreWarnings = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)