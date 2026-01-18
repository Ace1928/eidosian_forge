from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectApplicationAwareInterconnectBandwidthPercentagePolicy(_messages.Message):
    """A InterconnectApplicationAwareInterconnectBandwidthPercentagePolicy
  object.

  Fields:
    bandwidthPercentages: Specify bandwidth percentages for various traffic
      classes for queuing type Bandwidth Percent.
  """
    bandwidthPercentages = _messages.MessageField('InterconnectApplicationAwareInterconnectBandwidthPercentage', 1, repeated=True)