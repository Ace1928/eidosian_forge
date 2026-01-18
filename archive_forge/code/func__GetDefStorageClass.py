from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import shim_util
def _GetDefStorageClass(self):
    """Gets the default storage class for a bucket."""
    url_args = self.args
    some_matched = False
    for url_str in url_args:
        self._CheckIsGsUrl(url_str)
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['storageClass'])
        for blr in bucket_iter:
            some_matched = True
            print('%s: %s' % (blr.url_string.rstrip('/'), blr.root_object.storageClass))
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))