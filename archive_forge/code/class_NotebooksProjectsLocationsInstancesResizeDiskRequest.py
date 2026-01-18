from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesResizeDiskRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesResizeDiskRequest object.

  Fields:
    notebookInstance: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    resizeDiskRequest: A ResizeDiskRequest resource to be passed as the
      request body.
  """
    notebookInstance = _messages.StringField(1, required=True)
    resizeDiskRequest = _messages.MessageField('ResizeDiskRequest', 2)