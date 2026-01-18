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
def _RunSubCommand(self, func):
    try:
        self.sub_opts, self.args = getopt.getopt(self.args, self.command_spec.supported_sub_args)
        metrics.LogCommandParams(sub_opts=self.sub_opts)
        return func(self)
    except getopt.GetoptError:
        self.RaiseInvalidArgumentException()