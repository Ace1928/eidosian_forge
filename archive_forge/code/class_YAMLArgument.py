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
class YAMLArgument(object, metaclass=abc.ABCMeta):
    """Root for generating all arguments from yaml data.

  Requires all subclasses to contain Generate and Parse methods.
  """

    @classmethod
    def FromData(cls, data, api_version=None):
        group = data.get('group')
        if group:
            return ArgumentGroup.FromData(group, api_version)
        if data.get('resource_spec'):
            return YAMLConceptArgument.FromData(data, api_version)
        return Argument.FromData(data)

    @property
    @abc.abstractmethod
    def api_fields(self):
        """List of api fields this argument maps to."""

    @abc.abstractmethod
    def IsApiFieldSpecified(self, namespace):
        """Whether the argument with an api field is specified in the namespace."""

    @abc.abstractmethod
    def Generate(self, methods, shared_resource_flags):
        """Generates and returns the base argument."""

    @abc.abstractmethod
    def Parse(self, method, message, namespace, group_required):
        """Parses namespace for argument's value and appends value to req message."""