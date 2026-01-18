from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
class ServerTlsPoliciesCompleter(completers.MultiResourceCompleter):

    def __init__(self, **kwargs):
        super(ServerTlsPoliciesCompleter, self).__init__(completers=[GlobalServerTlsPoliciesCompleter, RegionServerTlsPoliciesCompleter], **kwargs)