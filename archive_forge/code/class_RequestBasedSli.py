from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestBasedSli(_messages.Message):
    """Service Level Indicators for which atomic units of service are counted
  directly.

  Fields:
    distributionCut: distribution_cut is used when good_service is a count of
      values aggregated in a Distribution that fall into a good range. The
      total_service is the total count of all values aggregated in the
      Distribution.
    goodTotalRatio: good_total_ratio is used when the ratio of good_service to
      total_service is computed from two TimeSeries.
  """
    distributionCut = _messages.MessageField('DistributionCut', 1)
    goodTotalRatio = _messages.MessageField('TimeSeriesRatio', 2)