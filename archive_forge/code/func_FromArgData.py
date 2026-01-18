from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
@classmethod
def FromArgData(cls, arg_data, resource_collection, is_list_method=False, shared_resource_args=None):
    if arg_data.repeated:
        gen_cls = UpdateListResourceArgumentGenerator
    else:
        gen_cls = UpdateDefaultResourceArgumentGenerator
    arg_name = arg_data.GetAnchorArgName(resource_collection, is_list_method)
    is_primary = arg_data.IsPrimaryResource(resource_collection)
    if is_primary:
        raise util.InvalidSchemaError('{} is a primary resource. Primary resources are required and cannot be listed as clearable.'.format(arg_name))
    api_field = _GetRelativeNameField(arg_data)
    if not api_field:
        raise util.InvalidSchemaError('{} does not specify the message field where the relative name is mapped in resource_method_params. Message field name is needed in order add update args. Please update resource_method_params.'.format(arg_name))
    return gen_cls(arg_name=arg_name, arg_gen=_GetResourceArgGenerator(arg_data, resource_collection, shared_resource_args), api_field=api_field, repeated=arg_data.repeated, collection=arg_data.collection, is_primary=is_primary, attribute_flags=arg_utils.GetAttributeFlags(arg_data, arg_name, resource_collection, shared_resource_args))