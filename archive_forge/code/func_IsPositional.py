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
def IsPositional(self, resource_collection=None, is_list_method=False):
    """Determines if the resource arg is positional.

    Args:
      resource_collection: APICollection | None, collection associated with
        the api method. None if a methodless command.
      is_list_method: bool | None, whether command is associated with list
        method. None if methodless command.

    Returns:
      bool, whether the resource arg anchor is positional
    """
    if self._is_positional is not None:
        return self._is_positional
    is_primary_resource = self.IsPrimaryResource(resource_collection)
    return is_primary_resource and (not is_list_method)