from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchBackgroundJobsResponse(_messages.Message):
    """Response message for 'SearchBackgroundJobs' request.

  Fields:
    jobs: The list of conversion workspace mapping rules.
  """
    jobs = _messages.MessageField('BackgroundJobLogEntry', 1, repeated=True)