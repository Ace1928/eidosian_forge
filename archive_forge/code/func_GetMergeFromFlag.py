from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetMergeFromFlag():
    """Anthos create-login-config Merge-From flag."""
    return base.Argument('--merge-from', required=False, help='Specifies the file path of an existing login configuration file to merge with.')