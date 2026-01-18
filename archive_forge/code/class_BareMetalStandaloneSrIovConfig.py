from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneSrIovConfig(_messages.Message):
    """Specifies the SR-IOV networking operator config.

  Fields:
    enabled: Whether to install the SR-IOV operator.
  """
    enabled = _messages.BooleanField(1)