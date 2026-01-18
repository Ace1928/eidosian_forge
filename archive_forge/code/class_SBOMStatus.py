from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SBOMStatus(_messages.Message):
    """The status of an SBOM generation.

  Enums:
    SbomStateValueValuesEnum: The progress of the SBOM generation.

  Fields:
    error: If there was an error generating an SBOM, this will indicate what
      that error was.
    sbomState: The progress of the SBOM generation.
  """

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
    error = _messages.StringField(1)
    sbomState = _messages.EnumField('SbomStateValueValuesEnum', 2)