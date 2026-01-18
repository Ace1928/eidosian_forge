from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfMetricTypeValueValuesEnum(_messages.Enum):
    """PerfMetricTypeValueValuesEnum enum type.

    Values:
      perfMetricTypeUnspecified: <no description>
      memory: <no description>
      cpu: <no description>
      network: <no description>
      graphics: <no description>
    """
    perfMetricTypeUnspecified = 0
    memory = 1
    cpu = 2
    network = 3
    graphics = 4