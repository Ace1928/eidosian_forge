from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def FleetMembershipAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='fleet_membership', help_text='attached cluster membership of the {resource}, in the form of projects/PROJECT/locations/global/memberships/MEMBERSHIP. ')