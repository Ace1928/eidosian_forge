from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
def _ProtocolAllowlist(self):
    allowlist = super(ListAlpha, self)._ProtocolAllowlist()
    return allowlist