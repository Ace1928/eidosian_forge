from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _UpdateGCSProfile(self, connection_profile, release_track, args, update_fields):
    """Updates GOOGLE CLOUD STORAGE connection profile."""
    if release_track == base.ReleaseTrack.BETA and args.IsSpecified('bucket_name'):
        connection_profile.gcsProfile.bucket = args.bucket_name
        update_fields.append('gcsProfile.bucket')
    if release_track == base.ReleaseTrack.GA and args.IsSpecified('bucket'):
        connection_profile.gcsProfile.bucket = args.bucket
        update_fields.append('gcsProfile.bucket')
    if args.IsSpecified('root_path'):
        connection_profile.gcsProfile.rootPath = args.root_path
        update_fields.append('gcsProfile.rootPath')