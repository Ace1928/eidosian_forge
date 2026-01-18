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
class CreateCommandGenerator(BaseCommandGenerator):
    """Generator for create commands."""
    command_type = yaml_command_schema.CommandType.CREATE

    def _Generate(self):
        """Generates a Create command.

    A create command has a single resource argument and an API to call to
    perform the creation. If the async section is given in the spec, an --async
    flag is added and polling is automatically done on the response. For APIs
    that adhere to standards, no further configuration is necessary. If the API
    uses custom operations, you may need to provide extra configuration to
    describe how to poll the operation.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """

        class Command(base.CreateCommand):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                if self.spec.async_:
                    base.ASYNC_FLAG.AddToParser(parser)
                if self.spec.arguments.labels:
                    labels_util.AddCreateLabelsFlags(parser)

            def Run(self_, args):
                ref, response = self._CommonRun(args)
                primary_resource_arg = self.arg_generator.GetPrimaryResource(self.methods, args).primary_resource
                is_parent_resource = primary_resource_arg and primary_resource_arg.is_parent_resource
                if self.spec.async_:
                    if ref is not None and (not is_parent_resource):
                        request_string = 'Create request issued for: [{{{}}}]'.format(yaml_command_schema_util.NAME_FORMAT_KEY)
                    else:
                        request_string = 'Create request issued'
                    response = self._HandleAsync(args, ref, response, request_string=request_string)
                    if args.async_:
                        return self._HandleResponse(response, args)
                if is_parent_resource:
                    response_obj = encoding.MessageToDict(response)
                    full_name = response_obj.get('response', {}).get('name')
                    if not full_name:
                        full_name = response_obj.get('name')
                    resource_name = resource_transform.TransformBaseName(full_name)
                else:
                    resource_name = self._GetDisplayName(ref, args)
                log.CreatedResource(resource_name, kind=self._GetDisplayResourceType(args))
                response = self._HandleResponse(response, args)
                return response
        return Command