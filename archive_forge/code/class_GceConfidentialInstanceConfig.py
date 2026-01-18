from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GceConfidentialInstanceConfig(_messages.Message):
    """A set of Compute Engine Confidential VM instance options.

  Fields:
    enableConfidentialCompute: Optional. Whether the instance has confidential
      compute enabled.
  """
    enableConfidentialCompute = _messages.BooleanField(1)