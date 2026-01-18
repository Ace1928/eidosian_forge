from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute.instance_groups.managed import wait_info
from googlecloudsdk.core import log
from googlecloudsdk.core.util import retry
class IgmState(enum.Enum):
    """Represents IGM state to wait for."""
    STABLE = 0
    VERSION_TARGET_REACHED = 1
    ALL_INSTANCES_CONFIG_EFFECTIVE = 2