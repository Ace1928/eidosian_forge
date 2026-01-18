from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscoderProjectsLocationsJobsCreateRequest(_messages.Message):
    """A TranscoderProjectsLocationsJobsCreateRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    parent: Required. The parent location to create and process this job.
      Format: `projects/{project}/locations/{location}`
  """
    job = _messages.MessageField('Job', 1)
    parent = _messages.StringField(2, required=True)