from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ResponsePolicyRuleAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='response-policy-rule', help_text='The Cloud DNS response policy rule name {resource}.')