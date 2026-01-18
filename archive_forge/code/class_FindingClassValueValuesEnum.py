from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FindingClassValueValuesEnum(_messages.Enum):
    """The class of the finding.

    Values:
      FINDING_CLASS_UNSPECIFIED: Unspecified finding class.
      THREAT: Describes unwanted or malicious activity.
      VULNERABILITY: Describes a potential weakness in software that increases
        risk to Confidentiality & Integrity & Availability.
      MISCONFIGURATION: Describes a potential weakness in cloud resource/asset
        configuration that increases risk.
      OBSERVATION: Describes a security observation that is for informational
        purposes.
      SCC_ERROR: Describes an error that prevents some SCC functionality.
      POSTURE_VIOLATION: Describes a potential security risk due to a change
        in the security posture.
    """
    FINDING_CLASS_UNSPECIFIED = 0
    THREAT = 1
    VULNERABILITY = 2
    MISCONFIGURATION = 3
    OBSERVATION = 4
    SCC_ERROR = 5
    POSTURE_VIOLATION = 6