from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.api_lib.workflows import poller_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.workflows import flags
from googlecloudsdk.core import resources
def ReleaseTrackToApiVersion(release_track):
    if release_track == base.ReleaseTrack.ALPHA:
        return 'v1alpha1'
    elif release_track == base.ReleaseTrack.BETA:
        return 'v1beta'
    elif release_track == base.ReleaseTrack.GA:
        return 'v1'
    else:
        raise UnsupportedReleaseTrackError(release_track)