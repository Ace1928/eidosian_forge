from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetLocationFlag():
    """Anthos location flag."""
    return base.Argument('--location', required=False, help='Specifies the Google Cloud location to use. If notspecified will use the current compute/zone property.')