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
def _GetRelativeNameField(arg_data):
    """Gets message field where the resource's relative name is mapped."""
    api_fields = [key for key, value in arg_data.resource_method_params.items() if util.REL_NAME_FORMAT_KEY in value]
    if not api_fields:
        return None
    return api_fields[0]