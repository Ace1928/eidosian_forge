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
def _ConfigureCommandArgumentParserArguments(self, parser, subcommands_or_arguments, gsutil_api):
    """Creates parsers recursively for potentially nested subcommands.

    Args:
      parser: argparse parser object.
      subcommands_or_arguments: list of CommandArgument objects, or recursive
          dict with subcommand names as keys.
      gsutil_api: gsutil Cloud API instance to use.

    Raises:
      RuntimeError: if argument is configured with unsupported completer
      TypeError: if subcommands_or_arguments is not a dict or list

    """
    logger = self._GetTabCompleteLogger()

    def HandleList():
        for command_argument in subcommands_or_arguments:
            action = parser.add_argument(*command_argument.args, **command_argument.kwargs)
            if command_argument.completer:
                action.completer = MakeCompleter(command_argument.completer, gsutil_api)

    def HandleDict():
        subparsers = parser.add_subparsers()
        for subcommand_name, subcommand_value in subcommands_or_arguments.items():
            cur_subcommand_parser = subparsers.add_parser(subcommand_name, add_help=False)
            logger.info('Constructing argument parsers for {}'.format(subcommand_name))
            self._ConfigureCommandArgumentParserArguments(cur_subcommand_parser, subcommand_value, gsutil_api)
    if isinstance(subcommands_or_arguments, list):
        HandleList()
    elif isinstance(subcommands_or_arguments, dict):
        HandleDict()
    else:
        error_format = 'subcommands_or_arguments {} should be list or dict, found type {}'
        raise TypeError(error_format.format(subcommands_or_arguments, type(subcommands_or_arguments)))