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
class GenericCommandGenerator(BaseCommandGenerator):
    """Generator for generic/custom commands."""
    command_type = yaml_command_schema.CommandType.GENERIC

    def _AddAsyncFlag(self, parser):
        if self.spec.async_:
            base.ASYNC_FLAG.AddToParser(parser)

    def _AddPagingFlags(self, parser):
        is_paginated = any((method.ListItemField() and method.HasTokenizedRequest() for method in self.methods))
        generic = self.spec.generic
        if not is_paginated or (generic and generic.disable_paging_flags):
            return
        base.FILTER_FLAG.AddToParser(parser)
        base.LIMIT_FLAG.AddToParser(parser)
        base.PAGE_SIZE_FLAG.AddToParser(parser)
        base.SORT_BY_FLAG.AddToParser(parser)
        if self.spec.response.id_field:
            base.URI_FLAG.AddToParser(parser)

    def _Generate(self):
        """Generates a generic command.

    A generic command has a resource argument, additional fields, and calls an
    API method. It supports async if the async configuration is given. Any
    fields is message_params will be generated as arguments and inserted into
    the request message.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """

        class Command(base.Command):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                self._AddAsyncFlag(parser)
                self._AddPagingFlags(parser)

            def Run(self_, args):
                self._RegisterURIFunc(args)
                ref, response = self._CommonRun(args)
                if self.spec.async_:
                    request_string = None
                    if ref:
                        request_string = 'Request issued for: [{{{}}}]'.format(yaml_command_schema_util.NAME_FORMAT_KEY)
                    response = self._HandleAsync(args, ref, response, request_string=request_string)
                return self._HandleResponse(response, args)
        return Command