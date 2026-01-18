from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArchitectureValueValuesEnum(_messages.Enum):
    """Output only. The CPU architecture for which packages in this
    distribution channel were built. Architecture will be blank for language
    packages.

    Values:
      ARCHITECTURE_UNSPECIFIED: Unknown architecture.
      X86: X86 architecture.
      X64: X64 architecture.
    """
    ARCHITECTURE_UNSPECIFIED = 0
    X86 = 1
    X64 = 2