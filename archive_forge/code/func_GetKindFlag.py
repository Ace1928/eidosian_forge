from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetKindFlag():
    return base.Argument('--kind', required=True, help='Policy type. Use `denypolicies` for deny policies.')