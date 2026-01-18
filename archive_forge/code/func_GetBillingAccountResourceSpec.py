from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetBillingAccountResourceSpec():
    return concepts.ResourceSpec('cloudcommerceconsumerprocurement.billingAccounts', resource_name='billing-account', billingAccountsId=BillingAccountAttributeConfig())