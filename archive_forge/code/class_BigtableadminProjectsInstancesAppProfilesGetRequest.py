from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesAppProfilesGetRequest(_messages.Message):
    """A BigtableadminProjectsInstancesAppProfilesGetRequest object.

  Fields:
    name: Required. The unique name of the requested app profile. Values are
      of the form
      `projects/{project}/instances/{instance}/appProfiles/{app_profile}`.
  """
    name = _messages.StringField(1, required=True)