from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectApplicationAwareInterconnectBandwidthPercentage(_messages.Message):
    """Specify bandwidth percentages (0-100) for various traffic classes in
  BandwidthPercentagePolicy. The sum of all percentages must equal 100. It is
  valid to specify percentages for some classes and not for others. The others
  will be implicitly marked as 0.

  Enums:
    TrafficClassValueValuesEnum: TrafficClass whose bandwidth percentage is
      being specified.

  Fields:
    percentage: Bandwidth percentage for a specific traffic class.
    trafficClass: TrafficClass whose bandwidth percentage is being specified.
  """

    class TrafficClassValueValuesEnum(_messages.Enum):
        """TrafficClass whose bandwidth percentage is being specified.

    Values:
      TC1: Traffic Class 1, corresponding to DSCP ranges (0-7) 000xxx.
      TC2: Traffic Class 2, corresponding to DSCP ranges (8-15) 001xxx.
      TC3: Traffic Class 3, corresponding to DSCP ranges (16-23) 010xxx.
      TC4: Traffic Class 4, corresponding to DSCP ranges (24-31) 011xxx.
      TC5: Traffic Class 5, corresponding to DSCP ranges (32-47) 10xxxx.
      TC6: Traffic Class 6, corresponding to DSCP ranges (48-63) 11xxxx.
    """
        TC1 = 0
        TC2 = 1
        TC3 = 2
        TC4 = 3
        TC5 = 4
        TC6 = 5
    percentage = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    trafficClass = _messages.EnumField('TrafficClassValueValuesEnum', 2)