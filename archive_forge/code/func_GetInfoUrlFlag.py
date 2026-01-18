from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetInfoUrlFlag():
    return base.Argument('--info-url', required=False, help='Url with more info about the package.')