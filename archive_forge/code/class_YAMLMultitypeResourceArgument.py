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
class YAMLMultitypeResourceArgument(YAMLConceptArgument):
    """Encapsulates the spec for the resource arg of a declarative command."""

    def __init__(self, data, group_help, request_api_version=None, **kwargs):
        super(YAMLMultitypeResourceArgument, self).__init__(data, group_help, **kwargs)
        self._resources = []
        for resource_data in data.get('resources', []):
            self._resources.append(YAMLResourceArgument.FromSpecData(resource_data, request_api_version, is_parent_resource=self.is_parent_resource))

    @property
    def collection(self):
        return None

    @property
    def _resource_spec(self):
        """Resource spec generated from the YAML."""
        resource_specs = []
        for sub_resource in self._resources:
            if not sub_resource._disable_auto_completers:
                raise ValueError('disable_auto_completers must be True for multitype resource argument [{}]'.format(self.name))
            resource_specs.append(sub_resource._resource_spec)
        return multitype.MultitypeResourceSpec(self.name, *resource_specs)

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

    def Generate(self, methods, shared_resource_flags=None):
        resource_collection = self._GetMethodCollection(methods)
        is_list_method = self._GetIsList(methods)
        return self.GenerateResourceArg(resource_collection, anchor_arg_name=self.GetAnchorArgName(resource_collection, is_list_method), shared_resource_flags=shared_resource_flags, group_help=self.group_help)

    def Parse(self, method, message, namespace, group_required=True):
        ref = self.ParseResourceArg(namespace, group_required)
        if not self.parse_resource_into_request or not ref:
            return message
        arg_utils.ParseResourceIntoMessage(ref, method, message, message_resource_map=self._GetResourceMap(ref), request_id_field=self.request_id_field, use_relative_name=self.use_relative_name, is_primary_resource=self.IsPrimaryResource(method and method.resource_argument_collection))