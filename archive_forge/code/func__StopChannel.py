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
def _StopChannel(self):
    channel_id = self.args[0]
    resource_id = self.args[1]
    self.logger.info('Removing channel %s with resource identifier %s ...', channel_id, resource_id)
    self.gsutil_api.StopChannel(channel_id, resource_id, provider='gs')
    self.logger.info('Succesfully removed channel.')
    return 0