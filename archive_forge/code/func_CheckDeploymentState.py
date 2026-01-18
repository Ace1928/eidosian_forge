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
def CheckDeploymentState(self, response):
    """Throws any unexpected states contained within deployment reponse.

    Args:
      response: run_apps.v1alpha1.deployment, response to check
    """
    dep_state = self.messages.DeploymentStatus.StateValueValuesEnum
    job_state = self.messages.JobDetails.StateValueValuesEnum
    if response.status.state == dep_state.SUCCEEDED:
        return
    if response.status.state == dep_state.FAILED:
        if not response.status.errorMessage:
            raise exceptions.IntegrationsOperationError('Configuration failed.')
        url = ''
        for job in response.status.jobDetails[::-1]:
            if job.state == job_state.FAILED:
                url = job.jobUri
                break
        error_msg = 'Configuration failed with error:\n  {}'.format('\n  '.join(response.status.errorMessage.split('; ')))
        if url:
            error_msg += '\nLogs are available at {}'.format(url)
        raise exceptions.IntegrationsOperationError(error_msg)
    else:
        raise exceptions.IntegrationsOperationError('Configuration returned in unexpected state "{}".'.format(response.status.state.name))