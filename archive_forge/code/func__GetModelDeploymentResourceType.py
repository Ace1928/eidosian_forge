from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.ai.models import client as model_client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.credentials import requests
from six.moves import http_client
def _GetModelDeploymentResourceType(model_ref, client, shared_resources_ref=None):
    """Gets the deployment resource type of a model.

  Args:
    model_ref: a model resource object.
    client: an apis.GetClientInstance object.
    shared_resources_ref: str, the shared deployment resource pool the model
      should use, formatted as the full URI

  Returns:
    A string which value must be 'DEDICATED_RESOURCES', 'AUTOMATIC_RESOURCES'
    or 'SHARED_RESOURCES'

  Raises:
    ArgumentError: if the model resource object is not found.
  """
    try:
        model_msg = model_client.ModelsClient(client=client).Get(model_ref)
    except apitools_exceptions.HttpError:
        raise errors.ArgumentError('There is an error while getting the model information. Please make sure the model %r exists.' % model_ref.RelativeName())
    model_resource = encoding.MessageToPyValue(model_msg)
    supported_deployment_resources_types = model_resource['supportedDeploymentResourcesTypes']
    if shared_resources_ref is not None:
        if 'SHARED_RESOURCES' not in supported_deployment_resources_types:
            raise errors.ArgumentError('Shared resources not supported for model {}.'.format(model_ref.RelativeName()))
        else:
            return 'SHARED_RESOURCES'
    try:
        supported_deployment_resources_types.remove('SHARED_RESOURCES')
        return supported_deployment_resources_types[0]
    except ValueError:
        return model_resource['supportedDeploymentResourcesTypes'][0]