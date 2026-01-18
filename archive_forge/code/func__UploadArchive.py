from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apigee import archives as cmd_lib
from googlecloudsdk.command_lib.apigee import defaults
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _UploadArchive(self, upload_url, zip_file_path):
    """Issues an HTTP PUT call to the upload URL with the zip file payload.

    Args:
      upload_url: A str containing the full upload URL.
      zip_file_path: A str of the local path to the zip file.

    Raises:
      googlecloudsdk.command_lib.apigee.errors.HttpRequestError if the response
        status of the HTTP PUT call is not 200 (OK).
    """
    upload_archive_resp = cmd_lib.UploadArchive(upload_url, zip_file_path)
    if not upload_archive_resp.ok:
        raise errors.HttpRequestError(upload_archive_resp.status_code, upload_archive_resp.reason, upload_archive_resp.content)