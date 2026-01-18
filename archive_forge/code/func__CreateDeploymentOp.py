from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _CreateDeploymentOp(deployment, deployment_full_name):
    """Initiates and returns a CreateDeployment operation.

  Args:
    deployment: A partially filled messages.Deployment. The deployment will be
      filled with other details before the operation is initiated.
    deployment_full_name: string, the fully qualified name of the deployment,
      e.g. "projects/p/locations/l/deployments/d".

  Returns:
    The CreateDeployment operation.
  """
    deployment_ref = resources.REGISTRY.Parse(deployment_full_name, collection='config.projects.locations.deployments')
    location_ref = deployment_ref.Parent()
    deployment_id = deployment_ref.Name()
    log.info('Creating the deployment')
    return configmanager_util.CreateDeployment(deployment, deployment_id, location_ref.RelativeName())