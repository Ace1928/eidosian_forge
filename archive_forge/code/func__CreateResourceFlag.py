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
def _CreateResourceFlag(self, flag_prefix=None, group_help=None):
    flag_name = arg_utils.GetFlagName(self.arg_name, flag_prefix=flag_prefix and flag_prefix.value)
    return self.arg_gen(flag_name, group_help=group_help)