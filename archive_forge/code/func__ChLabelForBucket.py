from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import codecs
import json
import os
import six
from gslib import metrics
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import LabelTranslation
@Retry(PreconditionException, tries=3, timeout_secs=1)
def _ChLabelForBucket(blr):
    url = blr.storage_url
    self.logger.info('Setting label configuration on %s...', blr)
    labels_message = None
    metageneration = None
    if self.gsutil_api.GetApiSelector(url.scheme) == ApiSelector.JSON:
        corrected_changes = self.label_changes
        if self.num_deletions:
            _, bucket_metadata = self.GetSingleBucketUrlFromArg(url.url_string, bucket_fields=['labels', 'metageneration'])
            if not bucket_metadata.labels:
                metageneration = bucket_metadata.metageneration
                corrected_changes = dict(((k, v) for k, v in six.iteritems(self.label_changes) if v))
        labels_message = LabelTranslation.DictToMessage(corrected_changes)
    else:
        _, bucket_metadata = self.GetSingleBucketUrlFromArg(url.url_string, bucket_fields=['labels', 'metageneration'])
        metageneration = bucket_metadata.metageneration
        label_json = {}
        if bucket_metadata.labels:
            label_json = json.loads(LabelTranslation.JsonFromMessage(bucket_metadata.labels))
        for key, value in six.iteritems(self.label_changes):
            if not value and key in label_json:
                del label_json[key]
            else:
                label_json[key] = value
        labels_message = LabelTranslation.DictToMessage(label_json)
    preconditions = Preconditions(meta_gen_match=metageneration)
    bucket_metadata = apitools_messages.Bucket(labels=labels_message)
    self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, preconditions=preconditions, provider=url.scheme, fields=['id'])