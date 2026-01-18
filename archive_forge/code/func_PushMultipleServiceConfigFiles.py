from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def PushMultipleServiceConfigFiles(service_name, config_files, is_async, validate_only=False, config_id=None):
    """Pushes a given set of service configuration files.

  Args:
    service_name: name of the service.
    config_files: a list of ConfigFile message objects.
    is_async: whether to wait for aync operations or not.
    validate_only: whether to perform a validate-only run of the operation
                     or not.
    config_id: an optional name for the config

  Returns:
    Full response from the SubmitConfigSource request.

  Raises:
    ServiceDeployErrorException: the SubmitConfigSource API call returned a
      diagnostic with a level of ERROR.
  """
    messages = GetMessagesModule()
    client = GetClientInstance()
    config_source = messages.ConfigSource(id=config_id)
    config_source.files.extend(config_files)
    config_source_request = messages.SubmitConfigSourceRequest(configSource=config_source, validateOnly=validate_only)
    submit_request = messages.ServicemanagementServicesConfigsSubmitRequest(serviceName=service_name, submitConfigSourceRequest=config_source_request)
    api_response = client.services_configs.Submit(submit_request)
    operation = ProcessOperationResult(api_response, is_async)
    response = operation.get('response', {})
    diagnostics = response.get('diagnostics', [])
    num_errors = 0
    for diagnostic in diagnostics:
        kind = diagnostic.get('kind', '').upper()
        logger = log.error if kind == 'ERROR' else log.warning
        msg = '{l}: {m}\n'.format(l=diagnostic.get('location'), m=diagnostic.get('message'))
        logger(msg)
        if kind == 'ERROR':
            num_errors += 1
    if num_errors > 0:
        exception_msg = '{0} diagnostic error{1} found in service configuration deployment. See log for details.'.format(num_errors, 's' if num_errors > 1 else '')
        raise exceptions.ServiceDeployErrorException(exception_msg)
    return response