from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def PromptToOverwriteCloud(args, settings, release_track):
    """If the service already exists, prompt the user before overwriting."""
    if ServiceExists(args, settings.project, settings.service_name, settings.region, release_track):
        if console_io.CanPrompt() and console_io.PromptContinue(message='Serivce {} already exists in project {}'.format(settings.service_name, settings.project), prompt_string='Do you want to overwrite it?'):
            return
        raise ServiceAlreadyExistsError('Service already exists.')