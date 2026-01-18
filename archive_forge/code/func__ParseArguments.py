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
def _ParseArguments(self, namespace, prefix, message):
    """Recursively generates data for the request message and any sub-messages.

    Args:
      namespace: The argparse namespace containing the all the parsed arguments.
      prefix: str, The flag prefix for the sub-message being generated.
      message: The apitools class for the message.

    Returns:
      A dict of message field data that can be passed to an apitools Message.
    """
    kwargs = {}
    for field in message.all_fields():
        arg_name = self._GetArgName(field.name)
        if not arg_name:
            continue
        arg_name = prefix + arg_name
        if field.variant == messages.Variant.MESSAGE:
            sub_kwargs = self._ParseArguments(namespace, arg_name + '.', field.type)
            if sub_kwargs:
                value = field.type(**sub_kwargs)
                kwargs[field.name] = value if not field.repeated else [value]
        else:
            value = arg_utils.GetFromNamespace(namespace, arg_name)
            if value is not None:
                kwargs[field.name] = arg_utils.ConvertValue(field, value)
    return kwargs