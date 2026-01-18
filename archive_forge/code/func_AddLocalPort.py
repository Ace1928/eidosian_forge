from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddLocalPort(self):
    self._AddFlag('--local-port', type=int, help='Local port to which the service connection is forwarded. If this flag is not set, then a random port is chosen.')