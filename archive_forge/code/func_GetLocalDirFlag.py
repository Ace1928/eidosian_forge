from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetLocalDirFlag(positional=True, required=True, help_override=None, metavar=None):
    """Get Local Package directory flag."""
    help_txt = help_override or '      The local directory to fetch the package to.\n      e.g. ./my-cockroachdb-copy\n      * If the directory does NOT exist: create the specified directory\n        and write the package contents to it\n\n      * If the directory DOES exist: create a NEW directory under the\n        specified one, defaulting the name to the Base of REPO/PKG_PATH\n\n      * If the directory DOES exist and already contains a directory with\n        the same name of the one that would be created: fail\n      '
    return GetFlagOrPositional(name='LOCAL_DIR', positional=positional, required=required, type=ExpandLocalDirAndVersion, help=help_txt, metavar=metavar)