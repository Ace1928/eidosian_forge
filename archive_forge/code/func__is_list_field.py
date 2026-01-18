from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
import six
@property
def _is_list_field(self):
    return self.field.name == arg_utils.ADDITIONAL_PROPS