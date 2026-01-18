from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VexAssessment(_messages.Message):
    """VexAssessment provides all publisher provided Vex information that is
  related to this vulnerability.

  Enums:
    StateValueValuesEnum: Provides the state of this Vulnerability assessment.

  Fields:
    cve: Holds the MITRE standard Common Vulnerabilities and Exposures (CVE)
      tracking number for the vulnerability. Deprecated: Use vulnerability_id
      instead to denote CVEs.
    impacts: Contains information about the impact of this vulnerability, this
      will change with time.
    justification: Justification provides the justification when the state of
      the assessment if NOT_AFFECTED.
    noteName: The VulnerabilityAssessment note from which this VexAssessment
      was generated. This will be of the form:
      `projects/[PROJECT_ID]/notes/[NOTE_ID]`.
    relatedUris: Holds a list of references associated with this vulnerability
      item and assessment.
    remediations: Specifies details on how to handle (and presumably, fix) a
      vulnerability.
    state: Provides the state of this Vulnerability assessment.
    vulnerabilityId: The vulnerability identifier for this Assessment. Will
      hold one of common identifiers e.g. CVE, GHSA etc.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Provides the state of this Vulnerability assessment.

    Values:
      STATE_UNSPECIFIED: No state is specified.
      AFFECTED: This product is known to be affected by this vulnerability.
      NOT_AFFECTED: This product is known to be not affected by this
        vulnerability.
      FIXED: This product contains a fix for this vulnerability.
      UNDER_INVESTIGATION: It is not known yet whether these versions are or
        are not affected by the vulnerability. However, it is still under
        investigation.
    """
        STATE_UNSPECIFIED = 0
        AFFECTED = 1
        NOT_AFFECTED = 2
        FIXED = 3
        UNDER_INVESTIGATION = 4
    cve = _messages.StringField(1)
    impacts = _messages.StringField(2, repeated=True)
    justification = _messages.MessageField('Justification', 3)
    noteName = _messages.StringField(4)
    relatedUris = _messages.MessageField('RelatedUrl', 5, repeated=True)
    remediations = _messages.MessageField('Remediation', 6, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    vulnerabilityId = _messages.StringField(8)