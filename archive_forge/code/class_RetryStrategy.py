from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RetryStrategy(_messages.Message):
    """The strategy for retrying failed patches during the patch window.

  Fields:
    enabled: If true, the agent will continue to try and patch until the
      window has ended.
  """
    enabled = _messages.BooleanField(1)