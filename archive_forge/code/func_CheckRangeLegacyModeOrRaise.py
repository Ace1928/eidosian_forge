from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def CheckRangeLegacyModeOrRaise(args):
    """Checks for range being used with incompatible mode and raises an error."""
    if args.IsSpecified('range') and args.IsSpecified('subnet_mode') and (args.subnet_mode != 'legacy'):
        raise parser_errors.ArgumentError(_RANGE_NON_LEGACY_MODE_ERROR)