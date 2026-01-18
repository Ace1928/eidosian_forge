from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateRepositoriesResponse(_messages.Message):
    """Message for response of creating repositories in batch.

  Fields:
    repositories: Repository resources created.
  """
    repositories = _messages.MessageField('Repository', 1, repeated=True)