from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HotTablet(_messages.Message):
    """A tablet is a defined by a start and end key and is explained in
  https://cloud.google.com/bigtable/docs/overview#architecture and
  https://cloud.google.com/bigtable/docs/performance#optimization. A Hot
  tablet is a tablet that exhibits high average cpu usage during the time
  interval from start time to end time.

  Fields:
    endKey: Tablet End Key (inclusive).
    endTime: Output only. The end time of the hot tablet.
    name: The unique name of the hot tablet. Values are of the form `projects/
      {project}/instances/{instance}/clusters/{cluster}/hotTablets/[a-zA-Z0-
      9_-]*`.
    nodeCpuUsagePercent: Output only. The average CPU usage spent by a node on
      this tablet over the start_time to end_time time range. The percentage
      is the amount of CPU used by the node to serve the tablet, from 0%
      (tablet was not interacted with) to 100% (the node spent all cycles
      serving the hot tablet).
    startKey: Tablet Start Key (inclusive).
    startTime: Output only. The start time of the hot tablet.
    tableName: Name of the table that contains the tablet. Values are of the
      form `projects/{project}/instances/{instance}/tables/_a-zA-Z0-9*`.
  """
    endKey = _messages.StringField(1)
    endTime = _messages.StringField(2)
    name = _messages.StringField(3)
    nodeCpuUsagePercent = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    startKey = _messages.StringField(5)
    startTime = _messages.StringField(6)
    tableName = _messages.StringField(7)