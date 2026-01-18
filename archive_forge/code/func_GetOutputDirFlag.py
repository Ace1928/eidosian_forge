from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetOutputDirFlag(positional=False, required=False, help_override=None, metavar='OUTPUT-DIR', default=None):
    """Anthos operation local output directory flag."""
    help_txt = help_override or 'The output directory of the cluster resources. If empty will export files to ./CLUSTER_NAME'
    return GetFlagOrPositional(name='OUTPUT_DIRECTORY', positional=positional, required=required, type=ExpandLocalDirAndVersion, help=help_txt, default=default, metavar=metavar)