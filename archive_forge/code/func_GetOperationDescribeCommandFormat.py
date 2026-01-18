from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def GetOperationDescribeCommandFormat(args):
    """Returns api version for the corresponding release track."""
    release_track = args.calliope_command.ReleaseTrack()
    if release_track == base.ReleaseTrack.ALPHA:
        return 'To check operation status, run: gcloud alpha compute os-config os-policy-assignments operations describe {}'
    elif release_track == base.ReleaseTrack.GA:
        return 'To check operation status, run: gcloud compute os-config os-policy-assignments operations describe {}'
    else:
        raise core_exceptions.UnsupportedReleaseTrackError(release_track)