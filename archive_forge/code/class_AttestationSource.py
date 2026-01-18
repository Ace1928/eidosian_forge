from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationSource(_messages.Message):
    """Specifies the locations for fetching the provenance attestations.

  Fields:
    containerAnalysisAttestationProjects: The IDs of the Google Cloud projects
      that store the SLSA attestations as Container Analysis Occurrences, in
      the format `projects/[PROJECT_ID]`. Maximum number of
      `container_analysis_attestation_projects` allowed in each
      `AttestationSource` is 10.
  """
    containerAnalysisAttestationProjects = _messages.StringField(1, repeated=True)