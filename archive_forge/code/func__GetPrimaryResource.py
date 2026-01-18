from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_property
def _GetPrimaryResource(resource_params, resource_collection):
    """Retrieves the primary resource arg.

  Args:
    resource_params: list of YAMLConceptParser
    resource_collection: registry.APICollection, resource collection
      associated with method

  Returns:
    YAMLConceptArgument (resource arg) or None.
  """
    if not resource_params:
        return None
    primary_resources = [arg for arg in resource_params if arg.IsPrimaryResource(resource_collection)]
    if not primary_resources:
        if resource_collection:
            full_name = resource_collection.full_name
            api_version = resource_collection.api_version
        else:
            full_name = None
            api_version = None
        raise util.InvalidSchemaError('No resource args were found that correspond with [{name} {version}]. Add resource arguments that corresponds with request.method collection [{name} {version}]. HINT: Can set resource arg is_primary_resource to True in yaml schema to receive more assistance with validation.'.format(name=full_name, version=api_version))
    if len(primary_resources) > 1:
        primary_resource_names = [arg.name for arg in primary_resources]
        raise util.InvalidSchemaError('Only one resource arg can be listed as primary. Remove one of the primary resource args [{}] or set is_primary_resource to False in yaml schema.'.format(', '.join(primary_resource_names)))
    return primary_resources[0]