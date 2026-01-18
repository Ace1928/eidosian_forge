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
def DeleteIntegration(self, name):
    """Delete an integration.

    Args:
      name:  str, the name of the resource to update.

    Raises:
      IntegrationNotFoundError: If the integration is not found.

    Returns:
      str, the type of the integration that is deleted.
    """
    app = self.GetDefaultApp()
    resource = self._FindResource(app.config, name)
    if resource is None:
        raise exceptions.IntegrationNotFoundError('Integration [{}] cannot be found'.format(name))
    try:
        typekit = typekits_util.GetTypeKitByResource(resource)
    except exceptions.ArgumentError:
        typekit = None
    bindings = base.BindingFinder(app.config.resourceList)
    binded_from_resources = bindings.GetIDsBindedTo(resource.id)
    unbind_match_type_names = []
    for rid in binded_from_resources:
        if rid.type == types_utils.SERVICE_TYPE:
            if self.GetCloudRunService(rid.name):
                unbind_match_type_names.append({'type': types_utils.SERVICE_TYPE, 'name': rid.name, 'ignoreResourceConfig': True})
    if typekit:
        delete_match_type_names = typekit.GetDeleteSelectors(name)
        resource_stages = typekit.GetDeleteComponentTypes(selectors=delete_match_type_names)
    else:
        delete_match_type_names = [{'type': resource.id.type, 'name': resource.id.name}]
        resource_stages = [resource.id.type]
    stages_map = stages.IntegrationDeleteStages(destroy_resource_types=resource_stages, should_configure_service=bool(unbind_match_type_names))

    def StatusUpdate(tracker, operation, unused_status):
        self._UpdateDeploymentTracker(tracker, operation, stages_map)
        return
    with progress_tracker.StagedProgressTracker('Deleting Integration...', stages_map.values(), failure_message='Failed to delete integration.') as tracker:
        if binded_from_resources:
            for rid in binded_from_resources:
                binded_res = self._FindResource(app.config, rid.name, rid.type)
                base.RemoveBinding(resource, binded_res)
            if unbind_match_type_names:
                self.ApplyAppConfig(tracker=tracker, tracker_update_func=StatusUpdate, appname=_DEFAULT_APP_NAME, appconfig=app.config, match_type_names=unbind_match_type_names, intermediate_step=True, etag=app.etag)
            else:
                self._UpdateApplication(appname=_DEFAULT_APP_NAME, appconfig=app.config, etag=app.etag)
        delete_selector = {'matchTypeNames': delete_match_type_names}
        self._UndeployResource(name, delete_selector, tracker, StatusUpdate)
    if typekit:
        return typekit.integration_type
    else:
        return resource.id.type