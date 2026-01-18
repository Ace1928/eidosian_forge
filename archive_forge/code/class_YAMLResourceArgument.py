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
class YAMLResourceArgument(YAMLConceptArgument):
    """Encapsulates the spec for the resource arg of a declarative command."""

    @classmethod
    def FromSpecData(cls, data, request_api_version, **kwargs):
        """Create a resource argument with no command-level information configured.

    Given just the reusable resource specification (such as attribute names
    and fallthroughs, it can be used to generate a ResourceSpec. Not suitable
    for adding directly to a command as a solo argument.

    Args:
      data: the yaml resource definition.
      request_api_version: str, api version of request collection.
      **kwargs: attributes outside of the resource spec

    Returns:
      YAMLResourceArgument with no group help or flag name information.
    """
        if not data:
            return None
        return cls(data, None, request_api_version=request_api_version, **kwargs)

    def __init__(self, data, group_help, request_api_version=None, **kwargs):
        super(YAMLResourceArgument, self).__init__(data, group_help, **kwargs)
        self._full_collection_name = data['collection']
        self._api_version = data.get('api_version', request_api_version)
        self.attribute_data = data['attributes']
        self._disable_auto_completers = data.get('disable_auto_completers', True)
        for removed in self.removed_flags:
            if removed not in self.attribute_names:
                raise util.InvalidSchemaError('Removed flag [{}] for resource arg [{}] references an attribute that does not exist. Valid attributes are [{}]'.format(removed, self.name, ', '.join(self.attribute_names)))

    @property
    def collection(self):
        return registry.GetAPICollection(self._full_collection_name, api_version=self._api_version)

    @property
    def _resource_spec(self):
        """Resource spec generated from the YAML."""
        attributes = concepts.ParseAttributesFromData(self.attribute_data, self.collection.detailed_params)
        return concepts.ResourceSpec(self.collection.full_name, resource_name=self.name, api_version=self.collection.api_version, disable_auto_completers=self._disable_auto_completers, plural_name=self._plural_name, is_positional=self._is_positional, **{attribute.parameter_name: attribute for attribute in attributes})

    def _GetParentResource(self, resource_collection):
        parent_collection, _, _ = resource_collection.full_name.rpartition('.')
        return registry.GetAPICollection(parent_collection, api_version=self._api_version)

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
        if not resource_collection or self.override_resource_collection:
            return True
        if self.is_parent_resource:
            resource_collection = self._GetParentResource(resource_collection)
        if resource_collection.full_name != self._full_collection_name:
            if self.is_primary_resource:
                raise util.InvalidSchemaError('Collection names do not match for resource argument specification [{}]. Expected [{}], found [{}]'.format(self.name, resource_collection.full_name, self._full_collection_name))
            return False
        if self._api_version and self._api_version != resource_collection.api_version:
            if self.is_primary_resource:
                raise util.InvalidSchemaError('API versions do not match for resource argument specification [{}]. Expected [{}], found [{}]'.format(self.name, resource_collection.api_version, self._api_version))
            return False
        return True

    def _GenerateUpdateFlags(self, resource_collection, is_list_method, shared_resource_flags=None):
        """Creates update flags generator using aptiools message."""
        return update_resource_args.UpdateResourceArgumentGenerator.FromArgData(self, resource_collection, is_list_method, shared_resource_flags)

    def _ParseUpdateArgsFromNamespace(self, resource_collection, is_list_method, namespace, message):
        """Parses update flags and returns modified apitools message field."""
        return self._GenerateUpdateFlags(resource_collection, is_list_method).Parse(namespace, message)

    def Generate(self, methods, shared_resource_flags=None):
        """Generates and returns resource argument.

    Args:
      methods: list[registry.APIMethod], used to generate other arguments.
      shared_resource_flags: [string], list of flags being generated elsewhere.

    Returns:
      Resource argument.
    """
        resource_collection = self._GetMethodCollection(methods)
        is_list_method = self._GetIsList(methods)
        if self.clearable:
            return self._GenerateUpdateFlags(resource_collection, is_list_method, shared_resource_flags).Generate()
        else:
            return self.GenerateResourceArg(resource_collection, anchor_arg_name=self.GetAnchorArgName(resource_collection, is_list_method), shared_resource_flags=shared_resource_flags, group_help=self.group_help)

    def Parse(self, method, message, namespace, group_required=True):
        """Sets the argument message value, if any, from the parsed args.

    Args:
      method: registry.APIMethod, used to parse other arguments.
      message: The API message, None for non-resource args.
      namespace: The parsed command line argument namespace.
      group_required: bool, whether parent argument group is required.
        Unused here.
    """
        if self.clearable:
            ref = self._ParseUpdateArgsFromNamespace(method and method.resource_argument_collection, method.IsList(), namespace, message)
        else:
            ref = self.ParseResourceArg(namespace, group_required)
        if not self.parse_resource_into_request or (not ref and (not self.clearable)):
            return
        arg_utils.ParseResourceIntoMessage(ref, method, message, message_resource_map=self._GetResourceMap(ref), request_id_field=self.request_id_field, use_relative_name=self.use_relative_name, is_primary_resource=self.IsPrimaryResource(method and method.resource_argument_collection))