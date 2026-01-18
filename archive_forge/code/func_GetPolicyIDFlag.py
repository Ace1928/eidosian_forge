from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetPolicyIDFlag():
    return base.Argument('policy_id', help='Policy ID that is unique for the resource to which the policy is attached.')