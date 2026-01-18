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
def _ParseResourcesForList(self, app: runapps_v1alpha1_messages.Application, integration_type_filter: str, service_name_filter: str, focus_workload: Optional[str]=None):
    """Helper function for ListIntegrations to parse relevant fields."""
    if app.config is None:
        return []
    app_resources = app.config.resourceList
    if not app_resources:
        return []
    bindings = base.BindingFinder(app_resources)
    deployment_cache = {}
    output = []
    for resource in sorted(app_resources, key=lambda x: x.id.name):
        try:
            typekit = typekits_util.GetTypeKitByResource(resource)
        except exceptions.ArgumentError:
            typekit = None
        integration_type = typekit.integration_type if typekit else resource.id.type
        if integration_type == focus_workload:
            continue
        if integration_type_filter and integration_type != integration_type_filter:
            continue
        if focus_workload:
            services = [res_id.name for res_id in bindings.GetBindingIDs(resource.id) if res_id.type == types_utils.SERVICE_TYPE]
        else:
            services = ['{}/{}'.format(res_id.type, res_id.name) for res_id in bindings.GetBindingIDs(resource.id)]
        if service_name_filter and service_name_filter not in services:
            continue
        status = self._GetStatusFromLatestDeployment(resource.latestDeployment, deployment_cache)
        region = app.name.split('/')[3]
        output.append(integration_list_printer.Row(integration_name=resource.id.name, region=region, integration_type=integration_type, services=','.join(sorted(services)), latest_deployment_status=six.text_type(status)))
    return output