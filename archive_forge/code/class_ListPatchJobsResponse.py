from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListPatchJobsResponse(_messages.Message):
    """A response message for listing patch jobs.

  Fields:
    nextPageToken: A pagination token that can be used to get the next page of
      results.
    patchJobs: The list of patch jobs.
  """
    nextPageToken = _messages.StringField(1)
    patchJobs = _messages.MessageField('PatchJob', 2, repeated=True)