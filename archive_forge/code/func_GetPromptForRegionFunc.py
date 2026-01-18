from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def GetPromptForRegionFunc(available_regions=constants.SUPPORTED_REGION):
    """Returns a no argument function that prompts available regions and catches the user selection."""
    return lambda: PromptForRegion(available_regions)