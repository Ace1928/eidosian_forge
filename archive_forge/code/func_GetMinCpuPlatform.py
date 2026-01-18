from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetMinCpuPlatform():
    """Gets the --min-cpu-platform flag."""
    return base.Argument('--min-cpu-platform', help='Optional minimum CPU platform of the reservation to create.')