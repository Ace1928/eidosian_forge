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
class DeleteCommandGenerator(BaseCommandGenerator):
    """Generator for delete commands."""
    command_type = yaml_command_schema.CommandType.DELETE

    def _Generate(self):
        """Generates a Delete command.

    A delete command has a single resource argument and an API to call to
    perform the delete. If the async section is given in the spec, an --async
    flag is added and polling is automatically done on the response. For APIs
    that adhere to standards, no further configuration is necessary. If the API
    uses custom operations, you may need to provide extra configuration to
    describe how to poll the operation.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """

        class Command(base.DeleteCommand):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                if self.spec.async_:
                    base.ASYNC_FLAG.AddToParser(parser)

            def Run(self_, args):
                ref, response = self._CommonRun(args)
                if self.spec.async_:
                    response = self._HandleAsync(args, ref, response, request_string='Delete request issued for: [{{{}}}]'.format(yaml_command_schema_util.NAME_FORMAT_KEY), extract_resource_result=False)
                    if args.async_:
                        return self._HandleResponse(response, args)
                response = self._HandleResponse(response, args)
                log.DeletedResource(self._GetDisplayName(ref, args), kind=self._GetDisplayResourceType(args))
                return response
        return Command