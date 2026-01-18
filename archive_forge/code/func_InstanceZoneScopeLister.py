from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def InstanceZoneScopeLister(compute_client, _, underspecified_names):
    """Scope lister for zones of underspecified instances."""
    messages = compute_client.messages
    instance_name = underspecified_names[0]
    project = properties.VALUES.core.project.Get(required=True)
    request = (compute_client.apitools_client.instances, 'AggregatedList', messages.ComputeInstancesAggregatedListRequest(filter='name eq ^{0}$'.format(instance_name), project=project, maxResults=constants.MAX_RESULTS_PER_PAGE))
    errors = []
    matching_instances = compute_client.MakeRequests([request], errors_to_collect=errors)
    zones = []
    if errors:
        log.debug('Errors fetching filtered aggregate list:\n{}'.format(errors))
        log.status.Print('Error fetching possible zones for instance: [{0}].'.format(', '.join(underspecified_names)))
        zones = zones_service.List(compute_client, project)
    elif not matching_instances:
        log.debug('Errors fetching filtered aggregate list:\n{}'.format(errors))
        log.status.Print('Unable to find an instance with name [{0}].'.format(instance_name))
        zones = zones_service.List(compute_client, project)
    else:
        for i in matching_instances:
            zone = core_resources.REGISTRY.Parse(i.zone, collection='compute.zones', params={'project': project})
            zones.append(messages.Zone(name=zone.Name()))
    return {compute_scope.ScopeEnum.ZONE: zones}