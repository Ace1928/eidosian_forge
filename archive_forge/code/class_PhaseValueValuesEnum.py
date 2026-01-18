from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PhaseValueValuesEnum(_messages.Enum):
    """Output only. The current migration job phase.

    Values:
      PHASE_UNSPECIFIED: The phase of the migration job is unknown.
      FULL_DUMP: The migration job is in the full dump phase.
      CDC: The migration job is CDC phase.
      PROMOTE_IN_PROGRESS: The migration job is running the promote phase.
      WAITING_FOR_SOURCE_WRITES_TO_STOP: Only RDS flow - waiting for source
        writes to stop
      PREPARING_THE_DUMP: Only RDS flow - the sources writes stopped, waiting
        for dump to begin
    """
    PHASE_UNSPECIFIED = 0
    FULL_DUMP = 1
    CDC = 2
    PROMOTE_IN_PROGRESS = 3
    WAITING_FOR_SOURCE_WRITES_TO_STOP = 4
    PREPARING_THE_DUMP = 5