from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobsDeleteRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobsDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the job is not found, the request will
      succeed but no action will be taken on the server.
    name: Required. The name of the job to delete. Format:
      `projects/{project}/locations/{location}/jobs/{job}`
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)