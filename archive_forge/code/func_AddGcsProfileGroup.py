from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddGcsProfileGroup(parser, release_track, required=True):
    """Adds necessary GCS profile flags to the given parser."""
    gcs_profile = parser.add_group()
    bucket_field_name = '--bucket'
    if release_track == base.ReleaseTrack.BETA:
        bucket_field_name = '--bucket-name'
    gcs_profile.add_argument(bucket_field_name, help='The full project and resource path for Cloud Storage\n      bucket including the name.', required=required)
    gcs_profile.add_argument('--root-path', help='The root path inside the Cloud Storage bucket.', required=False)