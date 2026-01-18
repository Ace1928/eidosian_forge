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
class YAMLConceptArgument(YAMLArgument, metaclass=abc.ABCMeta):
    """Encapsulate data used to generate and parse all resource args.

  YAMLConceptArgument is parent class that parses data and standardizes
  the interface (abstract base class) for YAML resource arguments by
  requiring methods Generate, Parse, and ParseResourceArg. All of the
  methods on YAMLConceptArgument are private helper methods for YAML
  resource arguments to share minor logic.
  """

    @classmethod
    def FromData(cls, data, api_version=None):
        if not data:
            return None
        resource_spec = data['resource_spec']
        help_text = data['help_text']
        kwargs = {'is_positional': data.get('is_positional'), 'is_parent_resource': data.get('is_parent_resource', False), 'is_primary_resource': data.get('is_primary_resource'), 'removed_flags': data.get('removed_flags'), 'arg_name': data.get('arg_name'), 'command_level_fallthroughs': data.get('command_level_fallthroughs', {}), 'display_name_hook': data.get('display_name_hook'), 'request_id_field': data.get('request_id_field'), 'resource_method_params': data.get('resource_method_params', {}), 'parse_resource_into_request': data.get('parse_resource_into_request', True), 'use_relative_name': data.get('use_relative_name', True), 'override_resource_collection': data.get('override_resource_collection', False), 'required': data.get('required'), 'repeated': data.get('repeated', False), 'request_api_version': api_version, 'clearable': data.get('clearable', False)}
        if 'resources' in data['resource_spec']:
            return YAMLMultitypeResourceArgument(resource_spec, help_text, **kwargs)
        else:
            return YAMLResourceArgument(resource_spec, help_text, **kwargs)

    def __init__(self, data, group_help, is_positional=None, removed_flags=None, is_parent_resource=False, is_primary_resource=None, arg_name=None, command_level_fallthroughs=None, display_name_hook=None, request_id_field=None, resource_method_params=None, parse_resource_into_request=True, use_relative_name=True, override_resource_collection=False, required=None, repeated=False, clearable=False, **unused_kwargs):
        self.flag_name_override = arg_name
        self.group_help = group_help
        self._is_positional = is_positional
        self.is_parent_resource = is_parent_resource
        self.is_primary_resource = is_primary_resource
        self.removed_flags = removed_flags or []
        self.command_level_fallthroughs = self._GenerateFallthroughsMap(command_level_fallthroughs)
        self.request_id_field = request_id_field or data.get('request_id_field')
        self.resource_method_params = resource_method_params or {}
        self.parse_resource_into_request = parse_resource_into_request
        self.use_relative_name = use_relative_name
        self.override_resource_collection = override_resource_collection
        self._required = required
        self.repeated = repeated
        self.clearable = clearable
        self.name = data['name']
        self._plural_name = data.get('plural_name')
        self.display_name_hook = util.Hook.FromPath(display_name_hook) if display_name_hook else None

    @property
    @abc.abstractmethod
    def _resource_spec(self):
        """"concepts.ConceptSpec generated from the YAML."""
        pass

    @property
    @abc.abstractmethod
    def collection(self):
        """"Get registry.APICollection based on collection and api_version."""
        pass

    @abc.abstractmethod
    def IsPrimaryResource(self, resource_collection):
        """Determines if this resource arg is the primary resource."""
        pass

    @property
    def attribute_names(self):
        """Names of resource attributes."""
        return [attr.name for attr in self._resource_spec.attributes]

    @property
    def api_fields(self):
        """Where the resource arg is mapped into the request message."""
        if self.resource_method_params:
            return list(self.resource_method_params.keys())
        else:
            return []

    @property
    def _anchor_name(self):
        """Name of the anchor attribute.

    For anchor attribute foo-bar, the expected format is...
      1. `foo-bar` if anchor is not positional
      2. `FOO_BAR` if anchor is positional
    """
        if self.flag_name_override:
            return self.flag_name_override
        else:
            count = 2 if self.repeated else 1
            return text.Pluralize(count, self._resource_spec.anchor.name)

    def GenerateResourceArg(self, method, anchor_arg_name=None, shared_resource_flags=None, group_help=None):
        """Generates only the resource arg (no update flags)."""
        return self._GenerateConceptParser(self._resource_spec, self.attribute_names, repeated=self.repeated, shared_resource_flags=shared_resource_flags, anchor_arg_name=anchor_arg_name, group_help=group_help, is_required=self.IsRequired(method))

    def ParseResourceArg(self, namespace, group_required=True):
        """Parses the resource ref from namespace (no update flags).

    Args:
      namespace: The argparse namespace.
      group_required: bool, whether parent argument group is required

    Returns:
      The parsed resource ref or None if no resource arg was generated for this
      method.
    """
        if not arg_utils.GetFromNamespace(namespace, self._anchor_name) and (not group_required):
            return None
        result = arg_utils.GetFromNamespace(namespace.CONCEPTS, self._anchor_name)
        if result:
            result = result.Parse()
        if isinstance(result, multitype.TypedConceptResult):
            return result.result
        else:
            return result

    def IsApiFieldSpecified(self, namespace):
        if not self.api_fields:
            return False
        return _IsSpecified(namespace=namespace, arg_dest=resource_util.NormalizeFormat(self._anchor_name), clearable=self.clearable)

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

    def IsRequired(self, resource_collection=None):
        """Determines if the resource arg is required.

    Args:
      resource_collection: APICollection | None, collection associated with
        the api method. None if a methodless command.

    Returns:
      bool, whether the resource arg is required
    """
        if self._required is not None:
            return self._required
        return self.IsPrimaryResource(resource_collection)

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

    def _GetMethodCollection(self, methods):
        for method in methods:
            if self.IsPrimaryResource(method.resource_argument_collection):
                return method.resource_argument_collection
        else:
            return methods[0].resource_argument_collection if methods else None

    def _GetIsList(self, methods):
        is_list = set((method.IsList() for method in methods))
        if len(is_list) > 1:
            raise util.InvalidSchemaError('Methods used to generate YAMLConceptArgument cannot contain both list and non-list methods. Update the list of methods to only use list or non-list methods.')
        if is_list:
            return is_list.pop()
        else:
            return False

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

    def _GenerateFallthroughsMap(self, command_level_fallthroughs_data):
        """Generate a map of command-level fallthroughs."""
        command_level_fallthroughs_data = command_level_fallthroughs_data or {}
        command_level_fallthroughs = {}

        def _FallthroughStringFromData(fallthrough_data):
            if fallthrough_data.get('is_positional', False):
                return resource_util.PositionalFormat(fallthrough_data['arg_name'])
            return resource_util.FlagNameFormat(fallthrough_data['arg_name'])
        for attr_name, fallthroughs_data in command_level_fallthroughs_data.items():
            fallthroughs_list = [_FallthroughStringFromData(fallthrough) for fallthrough in fallthroughs_data]
            command_level_fallthroughs[attr_name] = fallthroughs_list
        return command_level_fallthroughs

    def _GenerateConceptParser(self, resource_spec, attribute_names, repeated=False, shared_resource_flags=None, anchor_arg_name=None, group_help=None, is_required=False):
        """Generates a ConceptParser from YAMLConceptArgument.

    Args:
      resource_spec: concepts.ResourceSpec, used to create PresentationSpec
      attribute_names: names of resource attributes
      repeated: bool, whether or not the resource arg should be plural
      shared_resource_flags: [string], list of flags being generated elsewhere
      anchor_arg_name: string | None, anchor arg name
      group_help: string | None, group help text
      is_required: bool, whether the resource arg should be required

    Returns:
      ConceptParser that will be added to the parser.
    """
        shared_resource_flags = shared_resource_flags or []
        ignored_fields = list(concepts.IGNORED_FIELDS.values()) + self.removed_flags + shared_resource_flags
        no_gen = {n: '' for n in ignored_fields if n in attribute_names}
        command_level_fallthroughs = {}
        arg_fallthroughs = self.command_level_fallthroughs.copy()
        arg_fallthroughs.update({n: ['--' + n] for n in shared_resource_flags if n in attribute_names})
        concept_parsers.UpdateFallthroughsMap(command_level_fallthroughs, anchor_arg_name, arg_fallthroughs)
        presentation_spec_class = presentation_specs.ResourcePresentationSpec
        if isinstance(resource_spec, multitype.MultitypeResourceSpec):
            presentation_spec_class = presentation_specs.MultitypeResourcePresentationSpec
        return concept_parsers.ConceptParser([presentation_spec_class(anchor_arg_name, resource_spec, group_help=group_help, prefixes=False, required=is_required, flag_name_overrides=no_gen, plural=repeated)], command_level_fallthroughs=command_level_fallthroughs)