from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _ReleaseCommandPrefix(release_track):
    """Prefix for release track for printing commands.

  Args:
    release_track: Release track of the command being run.

  Returns:
    A formatted string of the release track prefix
  """
    track = release_track.prefix
    if track:
        track += ' '
    return track