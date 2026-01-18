from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def AddLoggingArg(parser):
    """Adds argument for specifying the logging level for an execution."""
    log_level = base.ChoiceArgument('--call-log-level', choices={'none': 'No logging level specified.', 'log-all-calls': 'Log all calls to subworkflows or library functions and their results.', 'log-errors-only': 'Log when a call is stopped due to an exception.', 'log-none': 'Perform no call logging.'}, help_str='Level of call logging to apply during execution.', default='none')
    log_level.AddToParser(parser)