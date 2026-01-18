from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def HasReplaceAdvertisementFlags(args):
    """Returns whether replace-style flags are specified in arguments."""
    return args.advertisement_mode or args.set_advertisement_groups is not None or args.set_advertisement_ranges is not None