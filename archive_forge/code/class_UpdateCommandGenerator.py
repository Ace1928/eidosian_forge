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
class UpdateCommandGenerator(BaseCommandGenerator):
    """Generator for update commands."""
    command_type = yaml_command_schema.CommandType.UPDATE

    def _Generate(self):
        """Generates an update command.

    An update command has a resource argument, additional fields, and calls an
    API method. It supports async if the async configuration is given. Any
    fields is message_params will be generated as arguments and inserted into
    the request message.

    Currently, the Update command is the same as Generic command.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.util.apis import update

        class Command(base.Command):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                if self.spec.async_:
                    base.ASYNC_FLAG.AddToParser(parser)
                if self.spec.arguments.labels:
                    labels_util.AddUpdateLabelsFlags(parser)

            def Run(self_, args):
                existing_message = None
                if self.spec.update:
                    if self.spec.update.read_modify_update:
                        existing_message = self._GetExistingResource(args)
                self.methods = self._GetRuntimeMethods(args)
                method = self.arg_generator.GetPrimaryResource(self.methods, args).method
                mask_path = update.GetMaskFieldPath(method)
                if mask_path:
                    if self.spec.update and self.spec.update.disable_auto_field_mask:
                        mask_string = ''
                    else:
                        mask_string = update.GetMaskString(args, self.spec, mask_path)
                    update_mask = {mask_path: mask_string}
                else:
                    update_mask = None
                ref, response = self._CommonRun(args, existing_message, update_mask)
                if self.spec.async_:
                    request_string = None
                    if ref:
                        request_string = 'Request issued for: [{{{}}}]'.format(yaml_command_schema_util.NAME_FORMAT_KEY)
                    response = self._HandleAsync(args, ref, response, request_string=request_string)
                log.UpdatedResource(self._GetDisplayName(ref, args), kind=self._GetDisplayResourceType(args))
                return self._HandleResponse(response, args)
        return Command