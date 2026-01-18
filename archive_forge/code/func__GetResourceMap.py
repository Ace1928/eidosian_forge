from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def _GetResourceMap(self, ref):
    message_resource_map = {}
    for message_field_name, param_str in self.resource_method_params.items():
        if ref is None:
            values = None
        elif isinstance(ref, list):
            values = [util.FormatResourceAttrStr(param_str, r) for r in ref]
        else:
            values = util.FormatResourceAttrStr(param_str, ref)
        message_resource_map[message_field_name] = values
    return message_resource_map