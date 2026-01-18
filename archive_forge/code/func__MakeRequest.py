from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.target_tcp_proxies import flags
def _MakeRequest(self, ref, holder):
    if ref.Collection() == 'compute.regionTargetTcpProxies':
        return self._MakeRegionalRequest(ref, holder)
    return self._MakeGlobalRequest(ref, holder)