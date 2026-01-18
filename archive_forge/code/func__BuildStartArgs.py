from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
def _BuildStartArgs(args):
    if platforms.OperatingSystem.Current() is not platforms.OperatingSystem.LINUX or args.use_docker:
        return _BuildStartArgsForDocker(args)
    else:
        return _BuildStartArgsForNativeExecutable(args)