from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetAwsStsRegionFlag():
    """Anthos auth token aws-sts-region flag, specifies the region for AWS STS endpoint for creating AWS token."""
    return base.Argument('--aws-sts-region', required=False, help='Region for AWS STS endpoint.')