from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Run services policies add-enable-rule.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The services in the consumer policy.
    