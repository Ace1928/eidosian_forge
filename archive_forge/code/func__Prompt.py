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
def _Prompt(self, parsed_args):
    """Fallthrough to reading the cluster location from an interactive prompt.

    Only prompt for cluster location if the user-specified platform is GKE
    and if cluster name is already defined.

    Args:
      parsed_args: Namespace, the args namespace.

    Returns:
      A cluster location string
    """
    cluster_name = getattr(parsed_args, 'cluster', None) or properties.VALUES.run.cluster.Get()
    if platforms.GetPlatform() == platforms.PLATFORM_GKE and cluster_name:
        clusters = [c for c in global_methods.ListClusters() if c.name == cluster_name]
        if not clusters:
            raise exceptions.ConfigurationError('No cluster locations found for cluster [{}]. Ensure your clusters have Cloud Run enabled.'.format(cluster_name))
        cluster_locations = [c.zone for c in clusters]
        idx = console_io.PromptChoice(cluster_locations, message='GKE cluster location for [{}]:'.format(cluster_name), cancel_option=True)
        location = cluster_locations[idx]
        log.status.Print('To make this the default cluster location, run `gcloud config set run/cluster_location {}`.\n'.format(location))
        return location