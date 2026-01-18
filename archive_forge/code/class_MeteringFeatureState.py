from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeteringFeatureState(_messages.Message):
    """Metering Feature State.

  Fields:
    lastMeasurementTime: The time stamp of the most recent measurement of the
      number of vCPUs in the cluster.
    preciseLastMeasuredClusterVcpuCapacity: The vCPUs capacity in the cluster
      according to the most recent measurement (1/1000 precision).
  """
    lastMeasurementTime = _messages.StringField(1)
    preciseLastMeasuredClusterVcpuCapacity = _messages.FloatField(2, variant=_messages.Variant.FLOAT)