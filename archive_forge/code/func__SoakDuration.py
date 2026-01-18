from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
def _SoakDuration(self) -> str:
    """Parses --soak-duration.

    Accepts ISO 8601 durations format. To read more,
    https://cloud.google.com/sdk/gcloud/reference/topic/

    Returns:
      str, in standard duration format, in unit of seconds.
    """
    if '--soak-duration' not in self.args.GetSpecifiedArgs():
        return None
    return '{}s'.format(self.args.soak_duration)