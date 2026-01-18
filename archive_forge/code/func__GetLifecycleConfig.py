from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import LifecycleTranslation
def _GetLifecycleConfig(self):
    """Gets lifecycle configuration for a Google Cloud Storage bucket."""
    bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['lifecycle'])
    if bucket_url.scheme == 's3':
        sys.stdout.write(self.gsutil_api.XmlPassThroughGetLifecycle(bucket_url, provider=bucket_url.scheme))
    elif bucket_metadata.lifecycle and bucket_metadata.lifecycle.rule:
        sys.stdout.write(LifecycleTranslation.JsonLifecycleFromMessage(bucket_metadata.lifecycle))
    else:
        sys.stdout.write('%s has no lifecycle configuration.\n' % bucket_url)
    return 0