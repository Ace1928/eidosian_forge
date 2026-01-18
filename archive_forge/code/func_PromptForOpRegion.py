from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def PromptForOpRegion():
    """Prompt for region from list of online prediction available regions.

  This method is referenced by the declaritive iam commands as a fallthrough
  for getting the region.

  Returns:
    The region specified by the user, str

  Raises:
    RequiredArgumentException: If can not prompt a console for region.
  """
    if console_io.CanPrompt():
        all_regions = list(constants.SUPPORTED_OP_REGIONS)
        idx = console_io.PromptChoice(all_regions, message='Please specify a region:\n', cancel_option=True)
        region = all_regions[idx]
        log.status.Print('To make this the default region, run `gcloud config set ai/region {}`.\n'.format(region))
        return region
    raise exceptions.RequiredArgumentException('--region', 'Cannot prompt a console for region. Region is required. Please specify `--region` to select a region.')