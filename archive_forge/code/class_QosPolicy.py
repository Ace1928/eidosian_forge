from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QosPolicy(_messages.Message):
    """QOS policy parameters.

  Fields:
    bandwidthGbps: The bandwidth permitted by the QOS policy, in gbps.
  """
    bandwidthGbps = _messages.FloatField(1)