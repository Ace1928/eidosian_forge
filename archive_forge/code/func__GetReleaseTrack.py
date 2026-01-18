from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
def _GetReleaseTrack(release_track):
    return 'v1alpha' if release_track == base.ReleaseTrack.ALPHA else 'v1'