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
def GetAnchorArgName(self, resource_collection, is_list_method):
    """Get the anchor argument name for the resource spec.

    Args:
      resource_collection: APICollection | None, collection associated with
        the api method. None if a methodless command.
      is_list_method: bool | None, whether command is associated with list
        method. None if methodless command.

    Returns:
      string, anchor in flag format ie `--foo-bar` or `FOO_BAR`
    """
    anchor_arg_is_flag = not self.IsPositional(resource_collection, is_list_method)
    return '--' + self._anchor_name if anchor_arg_is_flag else self._anchor_name