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
def GetPrimaryResource(self, methods, namespace):
    """Gets primary resource based on user input and returns single method.

    This determines which api method to use to make api request. If there
    is only one potential request method, return the one request method.

    Args:
      methods: list[APIMethod], The method to generate arguments for.
      namespace: The argparse namespace.

    Returns:
      MethodResourceArg, gets the primary resource arg and method the
        user specified in the namespace.

    Raises:
      ConflictingResourcesError: occurs when user specifies too many primary
        resources.
    """
    specified_methods = []
    primary_resources = _GetMethodResourceArgs(self.resource_args, methods)
    if not primary_resources:
        return MethodResourceArg(primary_resource=None, method=None)
    elif len(primary_resources) == 1:
        return primary_resources.pop()
    for method_info in primary_resources:
        method = method_info.method
        primary_resource = method_info.primary_resource
        if not method or not primary_resource:
            raise util.InvalidSchemaError('If more than one request collection is specified, a resource argument that corresponds with the collection, must be specified in YAML command.')
        method_collection = _GetCollectionName(method, is_parent=primary_resource.is_parent_resource)
        specified_resource = method_info.Parse(namespace)
        primary_collection = specified_resource and specified_resource.GetCollectionInfo().full_name
        if method_collection == primary_collection:
            specified_methods.append(method_info)
    if len(specified_methods) > 1:
        uris = []
        for method_info in specified_methods:
            if (parsed := method_info.Parse(namespace)):
                uris.append(parsed.RelativeName())
        args = ', '.join(uris)
        raise ConflictingResourcesError(f'User specified multiple primary resource arguments: [{args}]. Unable to determine api request method.')
    if len(specified_methods) == 1:
        return specified_methods.pop()
    else:
        return MethodResourceArg(primary_resource=None, method=None)