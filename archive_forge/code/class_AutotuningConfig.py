from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutotuningConfig(_messages.Message):
    """Autotuning configuration of the workload.

  Enums:
    ScenariosValueListEntryValuesEnum:

  Fields:
    cohort: Required. Autotuning cohort identifier. Identifies families of the
      workloads having the same shape, e.g. daily ETL jobs.
    scenarios: Required. Scenarios for which tunings are applied.
  """

    class ScenariosValueListEntryValuesEnum(_messages.Enum):
        """ScenariosValueListEntryValuesEnum enum type.

    Values:
      SCENARIO_UNSPECIFIED: Default value.
      OOM: Out-of-memory errors remediation.
      SCALING: Scaling recommendations such as initialExecutors.
      BHJ: Adding hints for potential relation broadcasts.
      MEMORY: Memory management for workloads.
    """
        SCENARIO_UNSPECIFIED = 0
        OOM = 1
        SCALING = 2
        BHJ = 3
        MEMORY = 4
    cohort = _messages.StringField(1)
    scenarios = _messages.EnumField('ScenariosValueListEntryValuesEnum', 2, repeated=True)