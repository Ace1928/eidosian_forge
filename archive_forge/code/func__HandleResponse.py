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
def _HandleResponse(self, response, args=None):
    """Process the API response.

    Args:
      response: The apitools message object containing the API response.
      args: argparse.Namespace, The parsed args.

    Raises:
      core.exceptions.Error: If an error was detected and extracted from the
        response.

    Returns:
      A possibly modified response.
    """
    if self.spec.response.error:
        error = self._FindPopulatedAttribute(response, self.spec.response.error.field.split('.'))
        if error:
            messages = []
            if self.spec.response.error.code:
                messages.append('Code: [{}]'.format(_GetAttribute(error, self.spec.response.error.code)))
            if self.spec.response.error.message:
                messages.append('Message: [{}]'.format(_GetAttribute(error, self.spec.response.error.message)))
            if messages:
                raise exceptions.Error(' '.join(messages))
            raise exceptions.Error(six.text_type(error))
    if self.spec.response.result_attribute:
        response = _GetAttribute(response, self.spec.response.result_attribute)
    for hook in self.spec.response.modify_response_hooks:
        response = hook(response, args)
    return response