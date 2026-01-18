from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def ListOccurrencesV1beta1(project, res_filter, page_size=1000):
    """List occurrences for resources in a project."""
    client = GetClientV1beta1()
    messages = GetMessagesV1beta1()
    project_ref = resources.REGISTRY.Parse(project, collection='cloudresourcemanager.projects')
    return list_pager.YieldFromList(client.projects_occurrences, request=messages.ContaineranalysisProjectsOccurrencesListRequest(parent=project_ref.RelativeName(), filter=res_filter), field='occurrences', batch_size=page_size, batch_size_attribute='pageSize')