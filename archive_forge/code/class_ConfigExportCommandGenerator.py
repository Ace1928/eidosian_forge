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
class ConfigExportCommandGenerator(BaseCommandGenerator):
    """Generator for config export commands."""
    command_type = yaml_command_schema.CommandType.CONFIG_EXPORT

    def _Generate(self):
        """Generates a config export command.

    A config export command has a resource argument as well as configuration
    export flags (such as --output-format and --path). It will export the
    configuration for one resource to stdout or to file, or will output a stream
    of configurations for all resources of the same type within a project to
    stdout, or to multiple files. Supported formats are `KRM` and `Terraform`.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.util.declarative import flags as declarative_config_flags
        from googlecloudsdk.command_lib.util.declarative import python_command_util

        class Command(base.Command):

            @staticmethod
            def Args(parser):
                mutex_group = parser.add_group(mutex=True, required=True)
                resource_group = mutex_group.add_group()
                args = self.arg_generator.GenerateArgs(self.methods)
                for arg in args:
                    for _, value in arg.specs.items():
                        value.required = False
                    arg.AddToParser(resource_group)
                declarative_config_flags.AddAllFlag(mutex_group, collection='project')
                declarative_config_flags.AddPathFlag(parser)
                declarative_config_flags.AddFormatFlag(parser)

            def Run(self_, args):
                resource_arg = self.arg_generator.GetPrimaryResource(self.methods, args).primary_resource
                collection = resource_arg and resource_arg.collection
                if getattr(args, 'all', None):
                    return python_command_util.RunExport(args=args, collection=collection.full_name, resource_ref=None)
                else:
                    return python_command_util.RunExport(args=args, collection=collection.full_name, resource_ref=resource_arg.ParseResourceArg(args).SelfLink())
        return Command