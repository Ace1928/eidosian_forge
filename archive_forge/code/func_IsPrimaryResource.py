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
def IsPrimaryResource(self, resource_collection):
    """Determines whether this resource arg is primary for a given method.

    Primary indicates that this resource arg represents the resource the api
    is fetching, updating, or creating

    Args:
      resource_collection: APICollection | None, collection associated with
        the api method. None if a methodless command.

    Returns:
      bool, true if this resource arg corresponds with the given method
        collection
    """
    if not self.is_primary_resource and self.is_primary_resource is not None:
        return False
    for sub_resource in self._resources:
        if sub_resource.IsPrimaryResource(resource_collection):
            return True
    if self.is_primary_resource:
        raise util.InvalidSchemaError('Collection names do not align with resource argument specification [{}]. Expected [{} version {}], and no contained resources matched.'.format(self.name, resource_collection.full_name, resource_collection.api_version))
    return False