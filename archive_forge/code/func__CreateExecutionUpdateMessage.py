from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateExecutionUpdateMessage(percent_complete, instance_details_json):
    """Constructs a message to be displayed during synchronous execute."""
    instance_states = {state: 0 for state in osconfig_command_utils.InstanceDetailsStates}
    for key, state in osconfig_command_utils.INSTANCE_DETAILS_KEY_MAP.items():
        num_instances = int(instance_details_json[key]) if key in instance_details_json else 0
        instance_states[state] = instance_states[state] + num_instances
    instance_details_str = ', '.join(['{} {}'.format(num, state.name.lower()) for state, num in instance_states.items()])
    return '{:.1f}% complete with {}'.format(percent_complete, instance_details_str)