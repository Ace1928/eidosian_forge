from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _configure_feature(self, parser):
    default_cfg = parser.load_fleet_default_cfg()
    if default_cfg is None:
        self.update_fleet_default(None)
    else:
        self.update_fleet_default(default_cfg)