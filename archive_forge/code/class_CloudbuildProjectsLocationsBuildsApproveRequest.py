from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBuildsApproveRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBuildsApproveRequest object.

  Fields:
    approveBuildRequest: A ApproveBuildRequest resource to be passed as the
      request body.
    name: Required. Name of the target build. For example:
      "projects/{$project_id}/builds/{$build_id}"
  """
    approveBuildRequest = _messages.MessageField('ApproveBuildRequest', 1)
    name = _messages.StringField(2, required=True)