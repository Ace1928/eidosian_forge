from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def ReleaseTrackCommandPrefix(release_track):
    """Returns a prefix to add to a gcloud command.

  This is meant for formatting an example string, such as:
    gcloud {}container fleet register-cluster

  Args:
    release_track: A ReleaseTrack

  Returns:
   a prefix to add to a gcloud based on the release track
  """
    prefix = release_track.prefix
    return prefix + ' ' if prefix else ''