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
def ConfigureCommandArgumentParsers(self, main_parser):
    """Configures argparse arguments and argcomplete completers for commands.

    Args:
      main_parser: argparse object that can be called to get subparsers to add
      subcommands (called just 'commands' in gsutil)
    """
    gsutil_api = self.GetGsutilApiForTabComplete()
    command_to_argparse_arguments = {command.command_spec.command_name: command.command_spec.argparse_arguments for command in self.command_map.values()}
    self._ConfigureCommandArgumentParserArguments(main_parser, command_to_argparse_arguments, gsutil_api)