from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesCopyInstanceRequest(_messages.Message):
    """A FileProjectsLocationsInstancesCopyInstanceRequest object.

  Fields:
    copyInstanceRequest: A CopyInstanceRequest resource to be passed as the
      request body.
    targetInstance: Required. The name of the Enterprise or High Scale
      instance instance that we are copying fileshare to, in the format `proje
      cts/{project_number}/locations/{location}/instances/{instance_id}`.
  """
    copyInstanceRequest = _messages.MessageField('CopyInstanceRequest', 1)
    targetInstance = _messages.StringField(2, required=True)