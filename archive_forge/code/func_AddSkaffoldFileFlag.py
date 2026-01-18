from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSkaffoldFileFlag():
    """Add --skaffold-file flag."""
    help_text = textwrap.dedent('  Path of the skaffold file absolute or relative to the source directory.\n\n  Examples:\n\n  Use Skaffold file with relative path:\n  The current working directory is expected to be some part of the skaffold path (e.g. the current working directory could be /home/user)\n\n    $ {command} --source=/home/user/source --skaffold-file=config/skaffold.yaml\n\n  The skaffold file absolute file path is expected to be:\n  /home/user/source/config/skaffold.yaml\n\n\n  Use Skaffold file with absolute path and with or without source argument:\n\n\n    $ {command} --source=/home/user/source --skaffold-file=/home/user/source/config/skaffold.yaml\n\n    $ {command} --skaffold-file=/home/user/source/config/skaffold.yaml\n\n  ')
    return base.Argument('--skaffold-file', help=help_text)