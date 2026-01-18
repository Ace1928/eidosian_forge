from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer.appliances import flags
def _get_gcs_destination_from_url_string(url_string):
    """Takes a storage_url string and returns a GcsDestination."""
    bucket, folder = _get_bucket_folder_from_url_string(url_string)
    return {'destination': {'outputBucket': bucket, 'outputPath': folder}}