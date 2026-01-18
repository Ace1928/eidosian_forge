from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCreateAppEngineTaskFlags(parser, is_alpha=False):
    """Add flags needed for creating a App Engine task to the parser."""
    AddQueueResourceFlag(parser, required=True)
    _GetTaskIdFlag().AddToParser(parser)
    flags = _AlphaAppEngineTaskFlags() if is_alpha else _AppEngineTaskFlags()
    for flag in flags:
        flag.AddToParser(parser)
    _AddPayloadFlags(parser, is_alpha)