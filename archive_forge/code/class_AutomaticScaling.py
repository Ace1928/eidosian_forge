from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomaticScaling(_messages.Message):
    """Automatic scaling is based on request rate, response latencies, and
  other application metrics.

  Fields:
    coolDownPeriod: The time period that the Autoscaler
      (https://cloud.google.com/compute/docs/autoscaler/) should wait before
      it starts collecting information from a new instance. This prevents the
      autoscaler from collecting information when the instance is
      initializing, during which the collected usage would not be reliable.
      Only applicable in the App Engine flexible environment.
    cpuUtilization: Target scaling by CPU usage.
    customMetrics: Target scaling by user-provided metrics. Only applicable in
      the App Engine flexible environment.
    diskUtilization: Target scaling by disk usage.
    maxConcurrentRequests: Number of concurrent requests an automatic scaling
      instance can accept before the scheduler spawns a new instance.Defaults
      to a runtime-specific value.
    maxIdleInstances: Maximum number of idle instances that should be
      maintained for this version.
    maxPendingLatency: Maximum amount of time that a request should wait in
      the pending queue before starting a new instance to handle it.
    maxTotalInstances: Maximum number of instances that should be started to
      handle requests for this version.
    minIdleInstances: Minimum number of idle instances that should be
      maintained for this version. Only applicable for the default version of
      a service.
    minPendingLatency: Minimum amount of time a request should wait in the
      pending queue before starting a new instance to handle it.
    minTotalInstances: Minimum number of running instances that should be
      maintained for this version.
    networkUtilization: Target scaling by network usage.
    requestUtilization: Target scaling by request utilization.
    standardSchedulerSettings: Scheduler settings for standard environment.
  """
    coolDownPeriod = _messages.StringField(1)
    cpuUtilization = _messages.MessageField('CpuUtilization', 2)
    customMetrics = _messages.MessageField('CustomMetric', 3, repeated=True)
    diskUtilization = _messages.MessageField('DiskUtilization', 4)
    maxConcurrentRequests = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    maxIdleInstances = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    maxPendingLatency = _messages.StringField(7)
    maxTotalInstances = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    minIdleInstances = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    minPendingLatency = _messages.StringField(10)
    minTotalInstances = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    networkUtilization = _messages.MessageField('NetworkUtilization', 12)
    requestUtilization = _messages.MessageField('RequestUtilization', 13)
    standardSchedulerSettings = _messages.MessageField('StandardSchedulerSettings', 14)