from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetCertificateMapEntryResourceSpec():
    return concepts.ResourceSpec('certificatemanager.projects.locations.certificateMaps.certificateMapEntries', resource_name='certificate map entry', certificateMapEntriesId=CertificateMapEntryAttributeConfig(), certificateMapsId=CertificateMapAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)