from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfidentialInstanceConfig(_messages.Message):
    """A set of Confidential Instance options.

  Fields:
    enableConfidentialCompute: Optional. Defines whether the instance should
      have confidential compute enabled.
  """
    enableConfidentialCompute = _messages.BooleanField(1)