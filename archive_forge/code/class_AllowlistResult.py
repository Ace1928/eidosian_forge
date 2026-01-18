from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowlistResult(_messages.Message):
    """Result of evaluating an image name allowlist.

  Fields:
    matchedPattern: The allowlist pattern that the image matched.
  """
    matchedPattern = _messages.StringField(1)