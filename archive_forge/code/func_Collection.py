from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workstations import configs
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.workstations import flags as workstations_flags
def Collection(self):
    return 'workstations.projects.locations.workstationClusters.workstationConfigs'