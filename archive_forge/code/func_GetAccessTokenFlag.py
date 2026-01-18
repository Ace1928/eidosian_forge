from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetAccessTokenFlag():
    """Anthos auth token access-token flag, specifies the Access Token received from identity provider after authorization flow."""
    return base.Argument('--access-token', required=False, help='Access Token received from identity provider after authorization flow.')