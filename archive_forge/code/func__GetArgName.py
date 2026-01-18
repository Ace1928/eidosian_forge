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
def _GetArgName(self, field_name, field_help=None):
    """Gets the name of the argument to generate for the field.

    Args:
      field_name: str, The name of the field.
      field_help: str, The help for the field in the API docs.

    Returns:
      str, The name of the argument to generate, or None if this field is output
      only or should be ignored.
    """
    if field_help and arg_utils.IsOutputField(field_help):
        return None
    if field_name in self.ignored_fields:
        return None
    if field_name == self.method.request_field and field_name.lower().endswith('request'):
        return 'request'
    return field_name