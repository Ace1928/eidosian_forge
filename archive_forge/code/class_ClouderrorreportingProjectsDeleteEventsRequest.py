from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouderrorreportingProjectsDeleteEventsRequest(_messages.Message):
    """A ClouderrorreportingProjectsDeleteEventsRequest object.

  Fields:
    projectName: Required. The resource name of the Google Cloud Platform
      project. Written as `projects/{projectID}`, where `{projectID}` is the
      [Google Cloud Platform project
      ID](https://support.google.com/cloud/answer/6158840). Example:
      `projects/my-project-123`.
  """
    projectName = _messages.StringField(1, required=True)