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
class WaitCommandGenerator(BaseCommandGenerator):
    """Generator for wait commands."""
    command_type = yaml_command_schema.CommandType.WAIT

    def _WaitForOperation(self, operation_ref, resource_ref, extract_resource_result, method, args=None):
        poller = AsyncOperationPoller(self.spec, resource_ref if extract_resource_result else None, args, operation_ref.GetCollectionInfo().full_name, method)
        return self._WaitForOperationWithPoller(poller, operation_ref, args=args)

    def _Generate(self):
        """Generates a wait command for polling operations.

    A wait command takes an operation reference and polls the status until it
    is finished or errors out. This follows the exact same spec as in other
    async commands except the primary operation (create, delete, etc) has
    already been done. For APIs that adhere to standards, no further async
    configuration is necessary. If the API uses custom operations, you may need
    to provide extra configuration to describe how to poll the operation.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """

        class Command(base.Command):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)

            def Run(self_, args):
                specified_resource = self.arg_generator.GetPrimaryResource(self.methods, args)
                method = specified_resource.method
                ref = specified_resource.Parse(args)
                response = self._WaitForOperation(ref, resource_ref=None, extract_resource_result=False, method=method, args=args)
                response = self._HandleResponse(response, args)
                return response
        return Command