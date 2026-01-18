from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
class ZonesCompleter(ResourceParamCompleter):
    """The zone completer."""

    def __init__(self, **kwargs):
        super(ZonesCompleter, self).__init__(collection='compute.zones', list_command='compute zones list --uri', param='zone', timeout=8 * 60 * 60, **kwargs)