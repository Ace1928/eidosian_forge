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
def _GenerateResourceArg(self):
    """Gets the flags to add to the parser that appear in the method path.

    Returns:
      {str, calliope.base.Argument}, A map of field name to argument.
    """
    args = []
    field_names = self.method.request_collection.detailed_params if self.method.request_collection else None
    if not field_names:
        return args
    field_helps = arg_utils.FieldHelpDocs(self.method.GetRequestType())
    default_help = 'For substitution into: ' + self.method.detailed_path
    arg = base.Argument(AutoArgumentGenerator.FLAT_RESOURCE_ARG_NAME, nargs='?', help='The GRI for the resource being operated on.')
    args.append(arg)
    for field in field_names:
        arg = base.Argument('--' + field, metavar=resource_property.ConvertToAngrySnakeCase(field), category='RESOURCE', help=field_helps.get(field, default_help))
        args.append(arg)
    return args