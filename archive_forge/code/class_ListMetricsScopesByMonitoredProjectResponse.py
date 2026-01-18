from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMetricsScopesByMonitoredProjectResponse(_messages.Message):
    """Response for the ListMetricsScopesByMonitoredProject method.

  Fields:
    metricsScopes: A set of all metrics scopes that the specified monitored
      project has been added to.
  """
    metricsScopes = _messages.MessageField('MetricsScope', 1, repeated=True)