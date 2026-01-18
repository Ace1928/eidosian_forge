from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def ConstructResourcePools(aiplatform_client, persistent_resource_config=None, resource_pool_specs=None, **kwargs):
    """Constructs the resource pools to be used to create a Persistent Resource.

  Resource pools from the config file and arguments will be combined.

  Args:
    aiplatform_client: The AI Platform API client used.
    persistent_resource_config: A Persistent Resource configuration imported
      from a YAML config.
    resource_pool_specs: A dict of worker pool specification, usually derived
      from the gcloud command argument values.
    **kwargs: The keyword args to pass to construct the worker pool specs.

  Returns:
    An array of ResourcePool messages for creating a Persistent Resource.
  """
    resource_pools = []
    if isinstance(persistent_resource_config.resourcePools, list):
        resource_pools = persistent_resource_config.resourcePools
    if resource_pool_specs:
        resource_pools = resource_pools + _ConstructResourcePoolSpecs(aiplatform_client, resource_pool_specs, **kwargs)
    return resource_pools