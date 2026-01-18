from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetSetPreferredAuthenticationFlag():
    return base.Argument('--set-preferred-auth', required=False, action='store_true', help='If set, forces update of preferred authentication for given cluster')