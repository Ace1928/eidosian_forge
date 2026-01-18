from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitServicePerimetersResponse(_messages.Message):
    """A response to CommitServicePerimetersRequest. This will be put inside of
  Operation.response field.

  Fields:
    servicePerimeters: List of all the Service Perimeter instances in the
      Access Policy.
  """
    servicePerimeters = _messages.MessageField('ServicePerimeter', 1, repeated=True)