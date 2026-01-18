from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilterValueValuesEnum(_messages.Enum):
    """Specify one or more PerfMetricType values such as CPU to filter the
    result

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