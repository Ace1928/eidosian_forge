from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import re
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_http as v2_docker_http
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from googlecloudsdk.api_lib.container.images import container_analysis_data_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.core.util import times
import six
from six.moves import map
import six.moves.http_client
def TransformManifests(manifests, repository, show_occurrences=False, occurrence_filter=filter_util.ContainerAnalysisFilter()):
    """Transforms the manifests returned from the server."""
    if not manifests:
        return []
    occurrences = {}
    if show_occurrences:
        occurrences = FetchOccurrences(repository, occurrence_filter=occurrence_filter)
    results = []
    for k, v in six.iteritems(manifests):
        result = {'digest': k, 'tags': v.get('tag', []), 'timestamp': _TimeCreatedToDateTime(v.get('timeCreatedMs'))}
        for occ in occurrences.get(_ResourceUrl(repository, k), []):
            if occ.kind not in result:
                result[occ.kind] = []
            result[occ.kind].append(occ)
        if show_occurrences and occurrence_filter.resources:
            result['vuln_counts'] = {}
            resource_url = _ResourceUrl(repository, k)
            if resource_url not in occurrence_filter.resources:
                continue
            summary = FetchSummary(repository, resource_url)
            for severity_count in summary.counts:
                if severity_count.severity:
                    result['vuln_counts'][str(severity_count.severity)] = severity_count.totalCount
        results.append(result)
    return results