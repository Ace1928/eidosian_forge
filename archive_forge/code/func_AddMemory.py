from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddMemory(self):
    self._AddFlag('--memory', type=arg_parsers.BinarySize(default_unit='B'), help='Container memory limit. Limit is expressed either as an integer representing the number of bytes or an integer followed by a unit suffix. Valid unit suffixes are "B", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB", or "PiB".')