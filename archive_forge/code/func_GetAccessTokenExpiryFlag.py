from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetAccessTokenExpiryFlag():
    """Anthos auth token access-token-expiry flag, specifies the Expiration time of access token received from identity provider after authorization flow."""
    return base.Argument('--access-token-expiry', required=False, help='Expiration time of access token received from identity provider after authorization flow. The expected format is the number of seconds elapsed since January 1, 1970 UTC.')