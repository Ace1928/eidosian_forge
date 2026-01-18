from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def CheckIncompatibleFlagsOrRaise(args):
    """Checks for incompatible flags in arguments and raises an error if found."""
    if HasReplaceAdvertisementFlags(args) and HasIncrementalAdvertisementFlags(args):
        raise parser_errors.ArgumentError(_INCOMPATIBLE_INCREMENTAL_FLAGS_ERROR_MESSAGE)