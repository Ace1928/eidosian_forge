from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Note(_messages.Message):
    """A type of analysis that can be done for a resource.

  Enums:
    KindValueValuesEnum: Output only. The type of analysis. This field can be
      used as a filter in list requests.

  Fields:
    attestation: A note describing an attestation role.
    build: A note describing build provenance for a verifiable build.
    compliance: A note describing a compliance check.
    createTime: Output only. The time this note was created. This field can be
      used as a filter in list requests.
    deployment: A note describing something that can be deployed.
    discovery: A note describing the initial analysis of a resource.
    dsseAttestation: A note describing a dsse attestation note.
    expirationTime: Time of expiration for this note. Empty if note does not
      expire.
    image: A note describing a base image.
    kind: Output only. The type of analysis. This field can be used as a
      filter in list requests.
    longDescription: A detailed description of this note.
    name: Output only. The name of the note in the form of
      `projects/[PROVIDER_ID]/notes/[NOTE_ID]`.
    package: A note describing a package hosted by various package managers.
    relatedNoteNames: Other notes related to this note.
    relatedUrl: URLs associated with this note.
    sbomReference: A note describing an SBOM reference.
    shortDescription: A one sentence description of this note.
    updateTime: Output only. The time this note was last updated. This field
      can be used as a filter in list requests.
    upgrade: A note describing available package upgrades.
    vulnerability: A note describing a package vulnerability.
    vulnerabilityAssessment: A note describing a vulnerability assessment.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Output only. The type of analysis. This field can be used as a filter
    in list requests.

    Values:
      NOTE_KIND_UNSPECIFIED: Default value. This value is unused.
      VULNERABILITY: The note and occurrence represent a package
        vulnerability.
      BUILD: The note and occurrence assert build provenance.
      IMAGE: This represents an image basis relationship.
      PACKAGE: This represents a package installed via a package manager.
      DEPLOYMENT: The note and occurrence track deployment events.
      DISCOVERY: The note and occurrence track the initial discovery status of
        a resource.
      ATTESTATION: This represents a logical "role" that can attest to
        artifacts.
      UPGRADE: This represents an available package upgrade.
      COMPLIANCE: This represents a Compliance Note
      DSSE_ATTESTATION: This represents a DSSE attestation Note
      VULNERABILITY_ASSESSMENT: This represents a Vulnerability Assessment.
      SBOM_REFERENCE: This represents an SBOM Reference.
    """
        NOTE_KIND_UNSPECIFIED = 0
        VULNERABILITY = 1
        BUILD = 2
        IMAGE = 3
        PACKAGE = 4
        DEPLOYMENT = 5
        DISCOVERY = 6
        ATTESTATION = 7
        UPGRADE = 8
        COMPLIANCE = 9
        DSSE_ATTESTATION = 10
        VULNERABILITY_ASSESSMENT = 11
        SBOM_REFERENCE = 12
    attestation = _messages.MessageField('AttestationNote', 1)
    build = _messages.MessageField('BuildNote', 2)
    compliance = _messages.MessageField('ComplianceNote', 3)
    createTime = _messages.StringField(4)
    deployment = _messages.MessageField('DeploymentNote', 5)
    discovery = _messages.MessageField('DiscoveryNote', 6)
    dsseAttestation = _messages.MessageField('DSSEAttestationNote', 7)
    expirationTime = _messages.StringField(8)
    image = _messages.MessageField('ImageNote', 9)
    kind = _messages.EnumField('KindValueValuesEnum', 10)
    longDescription = _messages.StringField(11)
    name = _messages.StringField(12)
    package = _messages.MessageField('PackageNote', 13)
    relatedNoteNames = _messages.StringField(14, repeated=True)
    relatedUrl = _messages.MessageField('RelatedUrl', 15, repeated=True)
    sbomReference = _messages.MessageField('SBOMReferenceNote', 16)
    shortDescription = _messages.StringField(17)
    updateTime = _messages.StringField(18)
    upgrade = _messages.MessageField('UpgradeNote', 19)
    vulnerability = _messages.MessageField('VulnerabilityNote', 20)
    vulnerabilityAssessment = _messages.MessageField('VulnerabilityAssessmentNote', 21)