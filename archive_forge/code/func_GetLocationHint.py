from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetLocationHint():
    """Gets the --location-hint flag."""
    return base.Argument('--location-hint', hidden=True, help='      Used by internal tools to control sub-zone location of the instance.\n      ')