from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveSeverityValueValuesEnum(_messages.Enum):
    """The distro assigned severity for this vulnerability when it is
    available, otherwise this is the note provider assigned severity. When
    there are multiple PackageIssues for this vulnerability, they can have
    different effective severities because some might be provided by the
    distro while others are provided by the language ecosystem for a language
    pack. For this reason, it is advised to use the effective severity on the
    PackageIssue level. In the case where multiple PackageIssues have
    differing effective severities, this field should be the highest severity
    for any of the PackageIssues.

    Values:
      SEVERITY_UNSPECIFIED: Unknown.
      MINIMAL: Minimal severity.
      LOW: Low severity.
      MEDIUM: Medium severity.
      HIGH: High severity.
      CRITICAL: Critical severity.
    """
    SEVERITY_UNSPECIFIED = 0
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5