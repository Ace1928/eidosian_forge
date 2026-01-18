from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildOperationMetadata(_messages.Message):
    """Metadata for build operations.

  Fields:
    build: The build that the operation is tracking.
  """
    build = _messages.MessageField('Build', 1)