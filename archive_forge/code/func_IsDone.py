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
def IsDone(self, operation):
    """Overrides."""
    result = getattr(operation, self.spec.async_.state.field)
    if isinstance(result, apitools_messages.Enum):
        result = result.name
    if result in self.spec.async_.state.success_values or result in self.spec.async_.state.error_values:
        error = getattr(operation, self.spec.async_.error.field)
        if not error and result in self.spec.async_.state.error_values:
            error = 'The operation failed.'
        if error:
            raise waiter.OperationError(SerializeError(error))
        return True
    return False