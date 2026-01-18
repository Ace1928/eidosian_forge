from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddDeleteTargetPositional(parser, folder):
    base.Argument('target', nargs='?', help='      A relative path to a file or subdirectory to delete within the\n      {folder} Cloud Storage subdirectory. If not specified, the entire contents\n      of the {folder} subdirectory will be deleted.\n      '.format(folder=folder)).AddToParser(parser)