from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
from six.moves import range
def MakeOccurrenceRequest(project_id, resource_filter, occurrence_filter=None, resource_urls=None):
    """Helper function to make Fetch Occurrence Request."""
    client = apis.GetClientInstance('containeranalysis', 'v1alpha1')
    messages = apis.GetMessagesModule('containeranalysis', 'v1alpha1')
    base_filter = resource_filter
    if occurrence_filter:
        base_filter = '({occurrence_filter}) AND ({base_filter})'.format(occurrence_filter=occurrence_filter, base_filter=base_filter)
    project_ref = resources.REGISTRY.Parse(project_id, collection='cloudresourcemanager.projects')
    if not resource_urls:
        return list_pager.YieldFromList(client.projects_occurrences, request=messages.ContaineranalysisProjectsOccurrencesListRequest(parent=project_ref.RelativeName(), filter=base_filter), field='occurrences', batch_size=1000, batch_size_attribute='pageSize')
    occurrence_generators = []
    for index in range(0, len(resource_urls), _MAXIMUM_RESOURCE_URL_CHUNK_SIZE):
        chunk = resource_urls[index:index + _MAXIMUM_RESOURCE_URL_CHUNK_SIZE]
        url_filter = '%s AND (%s)' % (base_filter, ' OR '.join(['resource_url="%s"' % url for url in chunk]))
        occurrence_generators.append(list_pager.YieldFromList(client.projects_occurrences, request=messages.ContaineranalysisProjectsOccurrencesListRequest(parent=project_ref.RelativeName(), filter=url_filter), field='occurrences', batch_size=1000, batch_size_attribute='pageSize'))
    return itertools.chain(*occurrence_generators)