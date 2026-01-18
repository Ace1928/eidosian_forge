from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimpleSigningAttestationCheck(_messages.Message):
    """Require a signed [DSSE](https://github.com/secure-systems-lab/dsse)
  attestation with type SimpleSigning.

  Fields:
    attestationAuthenticators: Required. The authenticators required by this
      check to verify an attestation. Typically this is one or more PKIX
      public keys for signature verification. Only one authenticator needs to
      consider an attestation verified in order for an attestation to be
      considered fully authenticated. In otherwords, this list of
      authenticators is an "OR" of the authenticator results. At least one
      authenticator is required.
    containerAnalysisAttestationProjects: Optional. The projects where
      attestations are stored as Container Analysis Occurrences, in the format
      `projects/[PROJECT_ID]`. Only one attestation needs to successfully
      verify an image for this check to pass, so a single verified attestation
      found in any of `container_analysis_attestation_projects` is sufficient
      for the check to pass. When fetching Occurrences from Container
      Analysis, only `AttestationOccurrence` kinds are considered. In the
      future, additional Occurrence kinds may be added to the query. Maximum
      number of `container_analysis_attestation_projects` allowed in each
      `SimpleSigningAttestationCheck` is 10.
  """
    attestationAuthenticators = _messages.MessageField('AttestationAuthenticator', 1, repeated=True)
    containerAnalysisAttestationProjects = _messages.StringField(2, repeated=True)