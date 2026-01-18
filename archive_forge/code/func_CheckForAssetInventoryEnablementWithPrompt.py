from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def CheckForAssetInventoryEnablementWithPrompt(project=None):
    """Checks if the cloudasset API is enabled, prompts to enable if not."""
    project = project or properties.VALUES.core.project.GetOrFail()
    service_name = 'cloudasset.googleapis.com'
    if not enable_api.IsServiceEnabled(project, service_name):
        if console_io.PromptContinue(default=False, prompt_string='API [{}] is required to continue, but is not enabled on project [{}]. Would you like to enable and retry (this will take a few minutes)?'.format(service_name, project)):
            enable_api.EnableService(project, service_name)
        else:
            raise AssetInventoryNotEnabledException('Aborted by user: API [{}] must be enabled on project [{}] to continue.'.format(service_name, project))