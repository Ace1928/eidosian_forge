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
def _WaitForOperationWithPoller(self, poller, operation_ref, args=None):
    progress_string = self._Format('Waiting for operation [{{{}}}] to complete'.format(yaml_command_schema_util.REL_NAME_FORMAT_KEY), operation_ref, self._GetDisplayResourceType(args))
    display_name = self._GetDisplayName(poller.resource_ref, args) if args else None
    return waiter.WaitFor(poller, operation_ref, self._Format(progress_string, poller.resource_ref, self._GetDisplayResourceType(args), display_name))