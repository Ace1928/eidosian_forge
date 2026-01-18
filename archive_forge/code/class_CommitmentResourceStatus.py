from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitmentResourceStatus(_messages.Message):
    """[Output Only] Contains output only fields.

  Fields:
    cancellationInformation: [Output Only] An optional, contains all the
      needed information of cancellation.
  """
    cancellationInformation = _messages.MessageField('CommitmentResourceStatusCancellationInformation', 1)