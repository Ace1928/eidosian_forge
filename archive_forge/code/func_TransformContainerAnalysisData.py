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
def TransformContainerAnalysisData(image_name, occurrence_filter=filter_util.ContainerAnalysisFilter()):
    """Transforms the occurrence data from Container Analysis API."""
    analysis_obj = container_analysis_data_util.ContainerAndAnalysisData(image_name)
    project_id = RecoverProjectId(image_name)
    occs = requests.ListOccurrences(project_id, occurrence_filter.GetFilter())
    for occ in occs:
        analysis_obj.add_record(occ)
    if 'DEPLOYMENT' in occurrence_filter.kinds:
        dep_filter = occurrence_filter.WithKinds(['DEPLOYMENT']).WithResources([])
        dep_occs = requests.ListOccurrences(project_id, dep_filter.GetFilter())
        image_string = six.text_type(image_name)
        for occ in dep_occs:
            if not occ.deployment:
                continue
            if image_string in occ.deployment.resourceUri:
                analysis_obj.add_record(occ)
    analysis_obj.resolveSummaries()
    return analysis_obj