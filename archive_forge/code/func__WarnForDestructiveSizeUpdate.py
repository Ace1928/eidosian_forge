from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def _WarnForDestructiveSizeUpdate(instance_ref, instance):
    """Adds prompt that warns about a destructive size update."""
    messages = util.GetMessagesForResource(instance_ref)
    message = 'Change to instance size requested. '
    if instance.tier == messages.Instance.TierValueValuesEnum.BASIC:
        message += 'Scaling a Basic Tier instance may result in data loss, and the instance will briefly be unavailable during the operation. '
    elif instance.tier == messages.Instance.TierValueValuesEnum.STANDARD_HA:
        message += 'Scaling a Standard Tier instance may result in the loss of unreplicated data, and the instance will be briefly unavailable during failover. '
    else:
        message += 'Scaling a redis instance may result in data loss, and the instance will be briefly unavailable during scaling. '
    message += 'For more information please take a look at https://cloud.google.com/memorystore/docs/redis/scaling-instances'
    console_io.PromptContinue(message=message, prompt_string='Do you want to proceed with update?', cancel_on_no=True)