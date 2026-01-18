from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.command_lib.container import container_command_util as cmd_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def AddFlags(channel, parser, flag_defaults, allowlist=None):
    """Adds flags to the current parser.

  Args:
    channel: channel from which to add flags. eg. "GA" or "BETA"
    parser: parser to add current flags to
    flag_defaults: mapping to override the default value of flags
    allowlist: only add intersection of this list and channel flags
  """
    add_flag_for_channel = flags_to_add[channel]
    for flagname in add_flag_for_channel:
        if allowlist is None or flagname in allowlist:
            if flagname in flag_defaults:
                add_flag_for_channel[flagname](parser, default=flag_defaults[flagname])
            else:
                add_flag_for_channel[flagname](parser)