from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def AddServiceVersionSelectArgs(parser, short_flags=False):
    """Add arguments to a parser for selecting service and version.

  Args:
    parser: An argparse.ArgumentParser.
    short_flags: bool, whether to add short flags `-s` and `-v` for service
      and version respectively.
  """
    parser.add_argument('--service', *(['-s'] if short_flags else []), required=False, help='The service ID.')
    parser.add_argument('--version', *(['-v'] if short_flags else []), required=False, help='The version ID.')