from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def HasIncrementalAdvertisementFlags(args):
    """Returns whether incremental-style flags are specified in arguments."""
    return args.add_advertisement_groups or args.remove_advertisement_groups or args.add_advertisement_ranges or args.remove_advertisement_ranges