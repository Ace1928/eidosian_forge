from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Assessment(_messages.Message):
    """Assessment provides all information that is related to a single
  vulnerability for this product.

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
    longDescription: A detailed description of this Vex.
    relatedUris: Holds a list of references associated with this vulnerability
      item and assessment. These uris have additional information about the
      vulnerability and the assessment itself. E.g. Link to a document which
      details how this assessment concluded the state of this vulnerability.
    remediations: Specifies details on how to handle (and presumably, fix) a
      vulnerability.
    shortDescription: A one sentence description of this Vex.
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
    longDescription = _messages.StringField(4)
    relatedUris = _messages.MessageField('RelatedUrl', 5, repeated=True)
    remediations = _messages.MessageField('Remediation', 6, repeated=True)
    shortDescription = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    vulnerabilityId = _messages.StringField(9)