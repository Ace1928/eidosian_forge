from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetClusterFlag(positional=False, required=False, help_override=None, metavar=None):
    """Anthos operation cluster name flag."""
    help_txt = help_override or 'Cluster to authenticate against. If no cluster is specified, the command will print a list of available options.'
    return GetFlagOrPositional(name='CLUSTER', positional=positional, required=required, help=help_txt, metavar=metavar)