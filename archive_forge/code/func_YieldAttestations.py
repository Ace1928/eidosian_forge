from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def YieldAttestations(self, note_ref=None, project_ref=None, artifact_digest=None, page_size=None, limit=None):
    """Yields occurrences associated with a given attestor Note or Project.

    Args:
      note_ref: The Note reference that will be queried for attached
        occurrences. If None, then all occurrences from the given project will
        be listed. (containeranalysis.projects.notes Resource)
      project_ref: The Project referenece that will be queried for occurrences
        if note_ref is None.
      artifact_digest: Digest of the artifact for which to fetch occurrences. If
        None, then all occurrences attached to the AA Note are returned.
      page_size: The number of attestations to retrieve per request. (If None,
        use the default page size.)
      limit: The maxium number of attestations to retrieve. (If None,
        unlimited.)

    Yields:
      Occurrences bound to `note_ref` with matching `artifact_digest` (if
      passed).
    """
    artifact_filter = 'has_suffix(resourceUrl, "{}")'.format(artifact_digest) if artifact_digest is not None else ''
    if note_ref is None:
        service = self.client.projects_occurrences
        request = self.messages.ContaineranalysisProjectsOccurrencesListRequest(parent=project_ref.RelativeName(), filter=artifact_filter)
    else:
        service = self.client.projects_notes_occurrences
        request = self.messages.ContaineranalysisProjectsNotesOccurrencesListRequest(name=note_ref.RelativeName(), filter=artifact_filter)
    occurrence_iter = list_pager.YieldFromList(service=service, request=request, field='occurrences', batch_size=page_size or 100, batch_size_attribute='pageSize', limit=limit)

    def MatchesFilter(occurrence):
        if occurrence.kind != self.messages.Occurrence.KindValueValuesEnum.ATTESTATION:
            return False
        return True
    for occurrence in occurrence_iter:
        if MatchesFilter(occurrence):
            yield occurrence