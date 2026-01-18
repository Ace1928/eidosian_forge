from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as http_exceptions
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files as file_utils
def SetFunctionSourceProps(function, function_ref, source_arg, stage_bucket, ignore_file=None, kms_key=None):
    """Add sources to function.

  Args:
    function: The function to add a source to.
    function_ref: The reference to the function.
    source_arg: Location of source code to deploy.
    stage_bucket: The name of the Google Cloud Storage bucket where source code
      will be stored.
    ignore_file: custom ignore_file name. Override .gcloudignore file to
      customize files to be skipped.
    kms_key: KMS key configured for the function.

  Returns:
    A list of fields on the function that have been changed.
  Raises:
    FunctionsError: If the kms_key doesn't exist or GCF P4SA lacks permissions.
  """
    function.sourceArchiveUrl = None
    function.sourceRepository = None
    function.sourceUploadUrl = None
    messages = api_util.GetApiMessagesModule()
    if source_arg is None:
        source_arg = '.'
    source_arg = source_arg or '.'
    if source_arg.startswith('gs://'):
        if not source_arg.endswith('.zip'):
            log.warning('[{}] does not end with extension `.zip`. The `--source` argument must designate the zipped source archive when providing a Google Cloud Storage URI.'.format(source_arg))
        function.sourceArchiveUrl = source_arg
        return ['sourceArchiveUrl']
    elif source_arg.startswith('https://'):
        function.sourceRepository = messages.SourceRepository(url=_AddDefaultBranch(source_arg))
        return ['sourceRepository']
    with file_utils.TemporaryDirectory() as tmp_dir:
        zip_file = source_util.CreateSourcesZipFile(tmp_dir, source_arg, ignore_file, enforce_size_limit=True)
        service = api_util.GetApiClientInstance().projects_locations_functions
        if stage_bucket:
            dest_object = source_util.UploadToStageBucket(zip_file, function_ref, stage_bucket)
            function.sourceArchiveUrl = dest_object.ToUrl()
            return ['sourceArchiveUrl']
        upload_url = _GetUploadUrl(messages, service, function_ref, kms_key)
        source_util.UploadToGeneratedUrl(zip_file, upload_url, extra_headers={'x-goog-content-length-range': '0,104857600'})
        function.sourceUploadUrl = upload_url
        return ['sourceUploadUrl']