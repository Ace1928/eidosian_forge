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
def GetIntegrationStatus(self, res_id: runapps_v1alpha1_messages.ResourceID) -> Optional[runapps_v1alpha1_messages.ResourceStatus]:
    """Get status of an integration.

    Args:
      res_id: ResourceID of the resource.

    Returns:
      The ResourceStatus of the integration, or None if not found
    """
    try:
        app_status = api_utils.GetApplicationStatus(self._client, self.GetAppRef(_DEFAULT_APP_NAME), [res_id])
        if app_status:
            for s in app_status.resourceStatuses:
                if s.id == res_id:
                    return s
        return None
    except api_exceptions.HttpError:
        return None