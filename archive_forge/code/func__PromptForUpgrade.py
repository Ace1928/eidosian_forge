from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
def _PromptForUpgrade(ref, args):
    """Prompts the user to confirm upgrade of instance."""
    scope_name = 'zone'
    resource_type = utils.CollectionToResourceType(ref.Collection())
    resource_name = utils.CamelCaseToOutputFriendly(resource_type)
    prompt_item = '[{0}] in [{1}]'.format(ref.Name(), getattr(ref, scope_name))
    prompt_title = 'The following {0} will be upgraded from {1} to {2}:'.format(resource_name, args.source_os, args.target_os)
    buf = io.StringIO()
    fmt = 'list[title="{title}",always-display-title]'.format(title=prompt_title)
    resource_printer.Print(prompt_item, fmt, out=buf)
    prompt_message = buf.getvalue()
    if not console_io.PromptContinue(message=prompt_message):
        raise compute_exceptions.AbortedError('Upgrade aborted by user.')