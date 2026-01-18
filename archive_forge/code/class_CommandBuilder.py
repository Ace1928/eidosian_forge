from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class CommandBuilder(object):
    """Generates calliope commands based on the yaml spec."""

    def __init__(self):
        self.command_generators = {}
        self.RegisterCommandGenerator(DescribeCommandGenerator)
        self.RegisterCommandGenerator(ListCommandGenerator)
        self.RegisterCommandGenerator(DeleteCommandGenerator)
        self.RegisterCommandGenerator(CreateCommandGenerator)
        self.RegisterCommandGenerator(WaitCommandGenerator)
        self.RegisterCommandGenerator(UpdateCommandGenerator)
        self.RegisterCommandGenerator(GenericCommandGenerator)
        self.RegisterCommandGenerator(GetIamPolicyCommandGenerator)
        self.RegisterCommandGenerator(SetIamPolicyCommandGenerator)
        self.RegisterCommandGenerator(AddIamPolicyBindingCommandGenerator)
        self.RegisterCommandGenerator(RemoveIamPolicyBindingCommandGenerator)
        self.RegisterCommandGenerator(ImportCommandGenerator)
        self.RegisterCommandGenerator(ExportCommandGenerator)
        self.RegisterCommandGenerator(ConfigExportCommandGenerator)

    def RegisterCommandGenerator(self, command_generator):
        if command_generator.command_type in self.command_generators:
            raise ValueError('Command type [{}] has already been registered.'.format(command_generator.command_type))
        self.command_generators[command_generator.command_type] = command_generator

    def GetCommandGenerator(self, spec, path):
        """Returns the command generator for a spec and path.

    Args:
      spec: yaml_command_schema.CommandData, the spec for the command being
        generated.
      path: Path for the command.

    Raises:
      ValueError: If we don't know how to generate the given command type (this
        is not actually possible right now due to the enum).

    Returns:
      The command generator.
    """
        if spec.command_type not in self.command_generators:
            raise ValueError('Command [{}] unknown command type [{}].'.format(' '.join(path), spec.command_type))
        return self.command_generators[spec.command_type](spec)

    def Generate(self, spec, path):
        """Generates a calliope command from the yaml spec.

    Args:
      spec: yaml_command_schema.CommandData, the spec for the command being
        generated.
      path: Path for the command.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        generator = self.GetCommandGenerator(spec, path)
        return generator.Generate()