from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import difflib
import logging
import os
import pkgutil
import sys
import textwrap
import time
import six
from six.moves import input
import boto
from boto import config
from boto.storage_uri import BucketStorageUri
import gslib
from gslib import metrics
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import GetFailureCount
from gslib.command import OLD_ALIAS_MAP
from gslib.command import ShutDownGsutil
import gslib.commands
from gslib.cs_api_map import ApiSelector
from gslib.cs_api_map import GsutilApiClassMapFactory
from gslib.cs_api_map import GsutilApiMapFactory
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.no_op_credentials import NoOpCredentials
from gslib.tab_complete import MakeCompleter
from gslib.utils import boto_util
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.utils.constants import RELEASE_NOTES_URL
from gslib.utils.constants import UTF8
from gslib.utils.metadata_util import IsCustomMetadataHeader
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.text_util import CompareVersions
from gslib.utils.text_util import InsistAsciiHeader
from gslib.utils.text_util import InsistAsciiHeaderValue
from gslib.utils.text_util import print_to_fd
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.update_util import LookUpGsutilVersion
from gslib.utils.update_util import GsutilPubTarball
def MaybeCheckForAndOfferSoftwareUpdate(self, command_name, debug):
    """Checks the last time we checked for an update and offers one if needed.

    Offer is made if the time since the last update check is longer
    than the configured threshold offers the user to update gsutil.

    Args:
      command_name: The name of the command being run.
      debug: Debug level to pass in to boto connection (range 0..3).

    Returns:
      True if the user decides to update.
    """
    logger = logging.getLogger()
    if self.SkipUpdateCheck() or command_name in ('config', 'update', 'ver', 'version') or system_util.InvokedViaCloudSdk():
        return False
    software_update_check_period = boto.config.getint('GSUtil', 'software_update_check_period', 30)
    if software_update_check_period == 0:
        return False
    last_checked_for_gsutil_update_timestamp_file = boto_util.GetLastCheckedForGsutilUpdateTimestampFile()
    cur_ts = int(time.time())
    if not os.path.isfile(last_checked_for_gsutil_update_timestamp_file):
        last_checked_ts = gslib.GetGsutilVersionModifiedTime()
        with open(last_checked_for_gsutil_update_timestamp_file, 'w') as f:
            f.write(str(last_checked_ts))
    else:
        try:
            with open(last_checked_for_gsutil_update_timestamp_file, 'r') as f:
                last_checked_ts = int(f.readline())
        except (TypeError, ValueError):
            return False
    if cur_ts - last_checked_ts > software_update_check_period * SECONDS_PER_DAY:
        gsutil_api = GcsJsonApi(self.bucket_storage_uri_class, logger, DiscardMessagesQueue(), credentials=NoOpCredentials(), debug=debug)
        cur_ver = gslib.VERSION
        try:
            cur_ver = LookUpGsutilVersion(gsutil_api, GsutilPubTarball())
        except Exception:
            return False
        with open(last_checked_for_gsutil_update_timestamp_file, 'w') as f:
            f.write(str(cur_ts))
        g, m = CompareVersions(cur_ver, gslib.VERSION)
        if m:
            print_to_fd('\n'.join(textwrap.wrap('A newer version of gsutil (%s) is available than the version you are running (%s). NOTE: This is a major new version, so it is strongly recommended that you review the release note details at %s before updating to this version, especially if you use gsutil in scripts.' % (cur_ver, gslib.VERSION, RELEASE_NOTES_URL))))
            if gslib.IS_PACKAGE_INSTALL:
                return False
            print_to_fd('\n')
            answer = input('Would you like to update [y/N]? ')
            return answer and answer.lower()[0] == 'y'
        elif g:
            print_to_fd('\n'.join(textwrap.wrap('A newer version of gsutil (%s) is available than the version you are running (%s). A detailed log of gsutil release changes is available at %s if you would like to read them before updating.' % (cur_ver, gslib.VERSION, RELEASE_NOTES_URL))))
            if gslib.IS_PACKAGE_INSTALL:
                return False
            print_to_fd('\n')
            answer = input('Would you like to update [Y/n]? ')
            return not answer or answer.lower()[0] != 'n'
    return False