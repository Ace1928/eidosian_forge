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
def _GetSharedAttributes(resource_params):
    """Retrieves shared attributes between resource args.

  Args:
    resource_params: [yaml_arg_schema.Argument], yaml argument tree

  Returns:
    Map of attribute names to list of resources that contain that attribute.
  """
    resource_names = set()
    attributes = {}
    for arg in resource_params:
        if arg.name in resource_names:
            if arg.name in resource_names and (not _DoesDupResourceArgHaveSameAttributes(arg, resource_params)):
                raise util.InvalidSchemaError('More than one resource argument has the name [{}] with different attributes. Remove the duplicate resource declarations.'.format(arg.name))
        else:
            resource_names.add(arg.name)
        for attribute_name in arg.attribute_names[:-1]:
            if attribute_name not in arg.removed_flags and (not concepts.IGNORED_FIELDS.get(attribute_name)):
                attributes[attribute_name] = attributes.get(attribute_name, [])
                attributes[attribute_name].append(arg.name)
    return {attribute: resource_args for attribute, resource_args in attributes.items() if len(resource_args) > 1}