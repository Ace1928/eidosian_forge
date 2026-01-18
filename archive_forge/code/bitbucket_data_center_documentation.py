from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import trigger_config as trigger_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
This is what gets called when the user runs this command.

    Args:
      args: An argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some values that we want to print later.
    