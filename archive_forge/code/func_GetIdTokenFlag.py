from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetIdTokenFlag():
    """Anthos auth token id-token flag, specifies the ID Token received from identity provider after authorization flow."""
    return base.Argument('--id-token', required=False, help='ID Token received from identity provider after authorization flow.')