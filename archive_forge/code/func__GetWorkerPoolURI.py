from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetWorkerPoolURI(resource):
    if isinstance(resource, dict):
        resource = resource['wp']
    wp = resources.REGISTRY.ParseRelativeName(resource.name, collection='cloudbuild.projects.locations.workerPools', api_version='v1')
    return wp.SelfLink()