from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def BytesToMb(size):
    """Converts a disk size in bytes to MB."""
    if not size:
        return None
    if size % constants.BYTES_IN_ONE_MB != 0:
        raise compute_exceptions.ArgumentError('Disk size must be a multiple of 1 MB. Did you mean [{0}MB]?'.format(size // constants.BYTES_IN_ONE_MB + 1))
    return size // constants.BYTES_IN_ONE_MB