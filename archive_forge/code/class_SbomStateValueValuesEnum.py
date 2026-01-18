from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SbomStateValueValuesEnum(_messages.Enum):
    """The progress of the SBOM generation.

    Values:
      SBOM_STATE_UNSPECIFIED: Default unknown state.
      PENDING: SBOM scanning is pending.
      COMPLETE: SBOM scanning has completed.
    """
    SBOM_STATE_UNSPECIFIED = 0
    PENDING = 1
    COMPLETE = 2