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
class ArgumentGroup(YAMLArgument):
    """Encapsulates data used to generate argument groups.

  Most of the attributes of this object correspond directly to the schema and
  have more complete docs there.

  Attributes:
    help_text: Optional help text for the group.
    required: True to make the group required.
    mutex: True to make the group mutually exclusive.
    hidden: True to make the group hidden.
    arguments: The list of arguments in the group.
  """

    @classmethod
    def FromData(cls, data, api_version=None):
        """Gets the arg group definition from the spec data.

    Args:
      data: The group spec data.
      api_version: Request method api version.

    Returns:
      ArgumentGroup, the parsed argument group.

    Raises:
      InvalidSchemaError: if the YAML command is malformed.
    """
        return cls(help_text=data.get('help_text'), required=data.get('required', False), mutex=data.get('mutex', False), hidden=data.get('hidden', False), arguments=[YAMLArgument.FromData(item, api_version) for item in data.get('params')])

    def __init__(self, help_text=None, required=False, mutex=False, hidden=False, arguments=None):
        super(ArgumentGroup, self).__init__()
        self.help_text = help_text
        self.required = required
        self.mutex = mutex
        self.hidden = hidden
        self.arguments = arguments

    @property
    def api_fields(self):
        api_fields = []
        for arg in self.arguments:
            api_fields.extend(arg.api_fields)
        return api_fields

    def IsApiFieldSpecified(self, namespace):
        for arg in self.arguments:
            if arg.IsApiFieldSpecified(namespace):
                return True
        else:
            return False

    def Generate(self, methods, shared_resource_flags=None):
        """Generates and returns the base argument group.

    Args:
      methods: list[registry.APIMethod], used to generate other arguments
      shared_resource_flags: [string], list of flags being generated elsewhere

    Returns:
      The base argument group.
    """
        group = base.ArgumentGroup(mutex=self.mutex, required=self.required, help=self.help_text, hidden=self.hidden)
        for arg in self.arguments:
            group.AddArgument(arg.Generate(methods, shared_resource_flags))
        return group

    def Parse(self, method, message, namespace, group_required=True):
        """Sets argument group message values, if any, from the parsed args.

    Args:
      method: registry.APIMethod, used to parse sub arguments.
      message: The API message, None for non-resource args.
      namespace: The parsed command line argument namespace.
      group_required: bool, if true, then parent argument group is required
    """
        arg_utils.ClearUnspecifiedMutexFields(message, namespace, self)
        for arg in self.arguments:
            arg.Parse(method, message, namespace, group_required and self.required)