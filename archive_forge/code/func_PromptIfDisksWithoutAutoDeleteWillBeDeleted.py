from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core.console import console_io
from six.moves import zip
def PromptIfDisksWithoutAutoDeleteWillBeDeleted(self, disks_to_warn_for):
    """Prompts if disks with False autoDelete will be deleted.

    Args:
      disks_to_warn_for: list of references to disk resources.
    """
    if not disks_to_warn_for:
        return
    prompt_list = []
    for ref in disks_to_warn_for:
        prompt_list.append('[{0}] in [{1}]'.format(ref.Name(), ref.zone))
    prompt_message = utils.ConstructList('The following disks are not configured to be automatically deleted with instance deletion, but they will be deleted as a result of this operation if they are not attached to any other instances:', prompt_list)
    if not console_io.PromptContinue(message=prompt_message):
        raise compute_exceptions.AbortedError('Deletion aborted by user.')