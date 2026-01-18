from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsDeleteRequest(_messages.Message):
    """A BigqueryJobsDeleteRequest object.

  Fields:
    jobId: Required. Job ID of the job for which metadata is to be deleted. If
      this is a parent job which has child jobs, the metadata from all child
      jobs will be deleted as well. Direct deletion of the metadata of child
      jobs is not allowed.
    location: The geographic location of the job. Required. See details at:
      https://cloud.google.com/bigquery/docs/locations#specifying_your_locatio
      n.
    projectId: Required. Project ID of the job for which metadata is to be
      deleted.
  """
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)