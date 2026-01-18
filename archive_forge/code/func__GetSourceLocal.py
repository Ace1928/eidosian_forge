import re
import types
from typing import FrozenSet, Optional, Tuple
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import types as api_types
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import labels_util
from googlecloudsdk.command_lib.functions import run_util
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
def _GetSourceLocal(args: parser_extensions.Namespace, client: base_api.BaseApiClient, function_ref: resources.Resource, source: str, kms_key: Optional[str]=None) -> api_types.Source:
    """Constructs a `Source` message from a local file system path.

  Args:
    args: The arguments that this command was invoked with.
    client: The GCFv2 Base API client.
    function_ref: The GCFv2 functions resource reference.
    source: The source path.
    kms_key: resource name of the customer managed KMS key | None

  Returns:
    The resulting cloudfunctions_v2_messages.Source.
  """
    messages = client.MESSAGES_MODULE
    with file_utils.TemporaryDirectory() as tmp_dir:
        zip_file_path = source_util.CreateSourcesZipFile(tmp_dir, source, args.ignore_file)
        if args.stage_bucket:
            dest_object = source_util.UploadToStageBucket(zip_file_path, function_ref, args.stage_bucket)
            return messages.Source(storageSource=messages.StorageSource(bucket=dest_object.bucket, object=dest_object.name))
        else:
            generate_upload_url_request = messages.GenerateUploadUrlRequest(kmsKeyName=kms_key)
            try:
                dest = client.projects_locations_functions.GenerateUploadUrl(messages.CloudfunctionsProjectsLocationsFunctionsGenerateUploadUrlRequest(generateUploadUrlRequest=generate_upload_url_request, parent=function_ref.Parent().RelativeName()))
            except apitools_exceptions.HttpError as e:
                cmek_util.ProcessException(e, kms_key)
                raise e
            source_util.UploadToGeneratedUrl(zip_file_path, dest.uploadUrl)
            return messages.Source(storageSource=dest.storageSource)