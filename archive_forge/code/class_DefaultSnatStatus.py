from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultSnatStatus(_messages.Message):
    """DefaultSnatStatus contains the desired state of whether default sNAT
  should be disabled on the cluster.

  Fields:
    disabled: Disables cluster default sNAT rules.
  """
    disabled = _messages.BooleanField(1)