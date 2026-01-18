from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_ex
from googlecloudsdk.api_lib.pubsub import schemas
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      A serialized object (dict) describing the results of the operation.
      This description fits the Resource described in the ResourceRegistry under
      'pubsub.projects.schemas'.

    Raises:
      util.RequestFailedError: if any of the requests to the API failed.
    