from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouderrorreportingProjectsGroupsGetRequest(_messages.Message):
    """A ClouderrorreportingProjectsGroupsGetRequest object.

  Fields:
    groupName: Required. The group resource name. Written as
      `projects/{projectID}/groups/{group_id}`. Call groupStats.list to return
      a list of groups belonging to this project. Example: `projects/my-
      project-123/groups/my-group` In the group resource name, the `group_id`
      is a unique identifier for a particular error group. The identifier is
      derived from key parts of the error-log content and is treated as
      Service Data. For information about how Service Data is handled, see
      [Google Cloud Privacy Notice](https://cloud.google.com/terms/cloud-
      privacy-notice).
  """
    groupName = _messages.StringField(1, required=True)