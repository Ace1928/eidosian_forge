from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def BillingAccountAttributeConfig(name=None, raw_help_text=None):
    if raw_help_text is not None:
        help_text = raw_help_text
    else:
        help_text = 'Cloud Billing account for the Procurement {resource}.'
    return concepts.ResourceParameterAttributeConfig(name=name if name is not None else 'billing-account', help_text=help_text)