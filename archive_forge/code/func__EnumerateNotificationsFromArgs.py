from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import re
import time
import uuid
from datetime import datetime
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PublishPermissionDeniedException
from gslib.command import Command
from gslib.command import NO_MAX
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.pubsub_apitools.pubsub_v1_messages import Binding
from gslib.utils import copy_helper
from gslib.utils import shim_util
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _EnumerateNotificationsFromArgs(self, accept_notification_configs=True):
    """Yields bucket/notification tuples from command-line args.

    Given a list of strings that are bucket names (gs://foo) or notification
    config IDs, yield tuples of bucket names and their associated notifications.

    Args:
      accept_notification_configs: whether notification configs are valid args.
    Yields:
      Tuples of the form (bucket_name, Notification)
    """
    path_regex = self._GetNotificationPathRegex()
    for list_entry in self.args:
        match = path_regex.match(list_entry)
        if match:
            if not accept_notification_configs:
                raise CommandException('%s %s accepts only bucket names, but you provided %s' % (self.command_name, self.subcommand_name, list_entry))
            bucket_name = match.group('bucket')
            notification_id = match.group('notification')
            found = False
            for notification in self.gsutil_api.ListNotificationConfigs(bucket_name, provider='gs'):
                if notification.id == notification_id:
                    yield (bucket_name, notification)
                    found = True
                    break
            if not found:
                raise NotFoundException('Could not find notification %s' % list_entry)
        else:
            storage_url = StorageUrlFromString(list_entry)
            if not storage_url.IsCloudUrl():
                raise CommandException('The %s command must be used on cloud buckets or notification config names.' % self.command_name)
            if storage_url.scheme != 'gs':
                raise CommandException('The %s command only works on gs:// buckets.')
            path = None
            if storage_url.IsProvider():
                path = 'gs://*'
            elif storage_url.IsBucket():
                path = list_entry
            if not path:
                raise CommandException('The %s command cannot be used on cloud objects, only buckets' % self.command_name)
            for blr in self.WildcardIterator(path).IterBuckets(bucket_fields=['id']):
                for notification in self.gsutil_api.ListNotificationConfigs(blr.storage_url.bucket_name, provider='gs'):
                    yield (blr.storage_url.bucket_name, notification)