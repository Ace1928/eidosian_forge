from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetUserFlag():
    return base.Argument('--user', required=False, help='If configuring multiple user accounts in the same kubecconfig file, you can specify a user to differentiate between them.')