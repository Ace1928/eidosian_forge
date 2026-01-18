from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetEtagFlag():
    return base.Argument('--etag', help='Etag that identifies the version of the existing policy. If omitted, the policy is deleted regardless of its current etag.')