from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def UpdateIntegration(self, name, parameters, add_service=None, remove_service=None):
    """Update an integration.

    Args:
      name:  str, the name of the resource to update.
      parameters: dict, the parameters from args.
      add_service: the service to attach to the integration.
      remove_service: the service to remove from the integration.

    Raises:
      IntegrationNotFoundError: If the integration is not found.

    Returns:
      The name of the integration.
    """
    app = self.GetDefaultApp()
    existing_resource = self._FindResource(app.config, name)
    if existing_resource is None:
        raise exceptions.IntegrationNotFoundError(messages_util.IntegrationNotFound(name))
    typekit = typekits_util.GetTypeKitByResource(existing_resource)
    flags.ValidateUpdateParameters(typekit.integration_type, parameters)
    specified_services = []
    services_in_params = typekit.UpdateResourceConfig(parameters, existing_resource)
    if services_in_params:
        specified_services.extend(services_in_params)
    if add_service:
        specified_services.append(add_service)
    match_type_names = typekit.GetCreateSelectors(name)
    for service in specified_services:
        self.EnsureWorkloadResources(app.config, service, types_utils.SERVICE_TYPE)
        self._AppendTypeMatcher(match_type_names, types_utils.SERVICE_TYPE, service, True)
    if add_service:
        workload = self._FindResource(app.config, add_service, types_utils.SERVICE_TYPE)
        typekit.BindServiceToIntegration(integration=existing_resource, workload=workload)
    if remove_service:
        workload_res = self._FindResource(app.config, remove_service, types_utils.SERVICE_TYPE)
        if workload_res:
            typekit.UnbindServiceFromIntegration(integration=existing_resource, workload=workload_res)
            if self.GetCloudRunService(remove_service):
                self._AppendTypeMatcher(match_type_names, types_utils.SERVICE_TYPE, remove_service)
        else:
            raise exceptions.ServiceNotFoundError('Service [{}] is not found among integrations'.format(remove_service))
    if specified_services:
        self.CheckCloudRunServicesExistence(specified_services)
    if typekit.is_ingress_service or (typekit.is_backing_service and add_service is None and (remove_service is None)):
        ref_svcs = typekit.GetBindedWorkloads(existing_resource, app.config.resourceList, types_utils.SERVICE_TYPE)
        for service in ref_svcs:
            if service not in specified_services and self.GetCloudRunService(service):
                self._AppendTypeMatcher(match_type_names, types_utils.SERVICE_TYPE, service, True)
    deploy_message = typekit.GetDeployMessage()
    resource_stages = typekit.GetCreateComponentTypes(selectors=match_type_names)
    stages_map = stages.IntegrationStages(create=False, resource_types=resource_stages)

    def StatusUpdate(tracker, operation, unused_status):
        self._UpdateDeploymentTracker(tracker, operation, stages_map)
        return
    with progress_tracker.StagedProgressTracker('Updating Integration...', stages_map.values(), failure_message='Failed to update integration.') as tracker:
        return self.ApplyAppConfig(tracker=tracker, tracker_update_func=StatusUpdate, appname=_DEFAULT_APP_NAME, appconfig=app.config, integration_name=name, deploy_message=deploy_message, match_type_names=match_type_names, etag=app.etag)