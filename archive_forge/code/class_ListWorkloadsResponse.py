from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkloadsResponse(_messages.Message):
    """Response to ListWorkloadsRequest.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    workloads: The list of environment workloads.
  """
    nextPageToken = _messages.StringField(1)
    workloads = _messages.MessageField('ComposerWorkload', 2, repeated=True)