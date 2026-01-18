from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def CreateDomainMapping(self, domain_mapping_ref, service_name, config_changes, force_override=False):
    """Create a domain mapping.

    Args:
      domain_mapping_ref: Resource, domainmapping resource.
      service_name: str, the service to which to map domain.
      config_changes: list of ConfigChanger to modify the domainmapping with
      force_override: bool, override an existing mapping of this domain.

    Returns:
      A domain_mapping.DomainMapping object.
    """
    messages = self.messages_module
    new_mapping = domain_mapping.DomainMapping.New(self._client, domain_mapping_ref.namespacesId)
    new_mapping.name = domain_mapping_ref.domainmappingsId
    new_mapping.route_name = service_name
    new_mapping.force_override = force_override
    for config_change in config_changes:
        new_mapping = config_change.Adjust(new_mapping)
    request = messages.RunNamespacesDomainmappingsCreateRequest(domainMapping=new_mapping.Message(), parent=domain_mapping_ref.Parent().RelativeName())
    with metrics.RecordDuration(metric_names.CREATE_DOMAIN_MAPPING):
        try:
            response = self._client.namespaces_domainmappings.Create(request)
        except api_exceptions.HttpConflictError:
            raise serverless_exceptions.DomainMappingCreationError('Domain mapping to [{}] already exists in this region.'.format(domain_mapping_ref.Name()))
        with progress_tracker.ProgressTracker('Creating...'):
            mapping = waiter.PollUntilDone(op_pollers.DomainMappingResourceRecordPoller(self), domain_mapping_ref)
        ready = mapping.conditions.get('Ready')
        message = None
        if ready and ready.get('message'):
            message = ready['message']
        if not mapping.records:
            if mapping.ready_condition['reason'] == domain_mapping.MAPPING_ALREADY_EXISTS_CONDITION_REASON:
                raise serverless_exceptions.DomainMappingAlreadyExistsError('Domain mapping to [{}] is already in use elsewhere.'.format(domain_mapping_ref.Name()))
            raise serverless_exceptions.DomainMappingCreationError(message or 'Could not create domain mapping.')
        if message:
            log.status.Print(message)
        return mapping
    return domain_mapping.DomainMapping(response, messages)