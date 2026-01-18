from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import os
import re
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util as concepts_util
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
class ClusterPromptFallthrough(PromptFallthrough):
    """Fall through to reading the cluster name from an interactive prompt."""

    def __init__(self):
        super(ClusterPromptFallthrough, self).__init__('specify the cluster from a list of available clusters')

    def _Prompt(self, parsed_args):
        """Fallthrough to reading the cluster name from an interactive prompt.

    Only prompt for cluster name if the user-specified platform is GKE.

    Args:
      parsed_args: Namespace, the args namespace.

    Returns:
      A cluster name string
    """
        if platforms.GetPlatform() != platforms.PLATFORM_GKE:
            return
        project = properties.VALUES.core.project.Get(required=True)
        cluster_location = getattr(parsed_args, 'cluster_location', None) or properties.VALUES.run.cluster_location.Get()
        cluster_location_msg = ' in [{}]'.format(cluster_location) if cluster_location else ''
        cluster_refs = global_methods.MultiTenantClustersForProject(project, cluster_location)
        if not cluster_refs:
            raise exceptions.ConfigurationError('No compatible clusters found{}. Ensure your cluster has Cloud Run enabled.'.format(cluster_location_msg))
        cluster_refs_descs = [self._GetClusterDescription(c, cluster_location, project) for c in cluster_refs]
        idx = console_io.PromptChoice(cluster_refs_descs, message='GKE cluster{}:'.format(cluster_location_msg), cancel_option=True)
        cluster_ref = cluster_refs[idx]
        if cluster_location:
            location_help_text = ''
        else:
            location_help_text = ' && gcloud config set run/cluster_location {}'.format(cluster_ref.zone)
        cluster_name = cluster_ref.Name()
        if cluster_ref.projectId != project:
            cluster_name = cluster_ref.RelativeName()
            location_help_text = ''
        log.status.Print('To make this the default cluster, run `gcloud config set run/cluster {cluster}{location}`.\n'.format(cluster=cluster_name, location=location_help_text))
        return cluster_ref.SelfLink()

    def _GetClusterDescription(self, cluster, cluster_location, project):
        """Description of cluster for prompt."""
        response = cluster.Name()
        if not cluster_location:
            response = '{} in {}'.format(response, cluster.zone)
        if project != cluster.projectId:
            response = '{} in {}'.format(response, cluster.projectId)
        return response