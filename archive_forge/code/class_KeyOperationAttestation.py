from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyOperationAttestation(_messages.Message):
    """Contains an HSM-generated attestation about a key operation. For more
  information, see [Verifying attestations]
  (https://cloud.google.com/kms/docs/attest-key).

  Enums:
    FormatValueValuesEnum: Output only. The format of the attestation data.

  Fields:
    certChains: Output only. The certificate chains needed to validate the
      attestation
    content: Output only. The attestation data provided by the HSM when the
      key operation was performed.
    format: Output only. The format of the attestation data.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Output only. The format of the attestation data.

    Values:
      ATTESTATION_FORMAT_UNSPECIFIED: Not specified.
      CAVIUM_V1_COMPRESSED: Cavium HSM attestation compressed with gzip. Note
        that this format is defined by Cavium and subject to change at any
        time. See https://www.marvell.com/products/security-solutions/nitrox-
        hs-adapters/software-key-attestation.html.
      CAVIUM_V2_COMPRESSED: Cavium HSM attestation V2 compressed with gzip.
        This is a new format introduced in Cavium's version 3.2-08.
    """
        ATTESTATION_FORMAT_UNSPECIFIED = 0
        CAVIUM_V1_COMPRESSED = 1
        CAVIUM_V2_COMPRESSED = 2
    certChains = _messages.MessageField('CertificateChains', 1)
    content = _messages.BytesField(2)
    format = _messages.EnumField('FormatValueValuesEnum', 3)