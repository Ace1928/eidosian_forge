from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsPatchRequest(_messages.Message):
    """A RunProjectsLocationsJobsPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, and if the Job does not exist, it
      will create a new one. Caller must have both create and update
      permissions for this call if this is set to true.
    googleCloudRunV2Job: A GoogleCloudRunV2Job resource to be passed as the
      request body.
    name: The fully qualified name of this Job. Format:
      projects/{project}/locations/{location}/jobs/{job}
    validateOnly: Indicates that the request should be validated and default
      values populated, without persisting the request or updating any
      resources.
  """
    allowMissing = _messages.BooleanField(1)
    googleCloudRunV2Job = _messages.MessageField('GoogleCloudRunV2Job', 2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)