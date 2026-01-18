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
def _GenerateArguments(self, prefix, message):
    """Gets the arguments to add to the parser that appear in the method body.

    Args:
      prefix: str, A string to prepend to the name of the flag. This is used
        for flags representing fields of a submessage.
      message: The apitools message to generate the flags for.

    Returns:
      {str, calliope.base.Argument}, A map of field name to argument.
    """
    args = []
    field_helps = arg_utils.FieldHelpDocs(message)
    for field in message.all_fields():
        field_help = field_helps.get(field.name, None)
        name = self._GetArgName(field.name, field_help)
        if not name:
            continue
        name = prefix + name
        if field.variant == messages.Variant.MESSAGE:
            sub_args = self._GenerateArguments(name + '.', field.type)
            if sub_args:
                help_text = name + ': ' + field_help if field_help else ''
                group = base.ArgumentGroup(help=help_text)
                args.append(group)
                for arg in sub_args:
                    group.AddArgument(arg)
        else:
            attributes = yaml_arg_schema.Argument(name, name, field_help)
            arg = arg_utils.GenerateFlag(field, attributes, fix_bools=False, category='MESSAGE')
            if not arg.kwargs.get('help'):
                arg.kwargs['help'] = 'API doc needs help for field [{}].'.format(name)
            args.append(arg)
    return args