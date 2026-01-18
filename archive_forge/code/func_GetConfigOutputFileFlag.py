from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetConfigOutputFileFlag():
    """Anthos create-login-config output flag."""
    return base.Argument('--output', required=False, type=ExpandLocalDirAndVersion, help='Destination to write login configuration file. Defaults to "kubectl-anthos-config.yaml".')