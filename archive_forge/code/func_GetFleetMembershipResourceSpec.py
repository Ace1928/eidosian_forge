from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetFleetMembershipResourceSpec():
    return concepts.ResourceSpec('gkehub.projects.locations.memberships', resource_name='fleet_membership', locationsId=LocationAttributeConfig(), membershipsId=FleetMembershipAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)