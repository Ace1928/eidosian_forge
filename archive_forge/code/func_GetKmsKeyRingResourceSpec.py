from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetKmsKeyRingResourceSpec(kms_prefix=True, region_fallthrough=False):
    return concepts.ResourceSpec('cloudkms.projects.locations.keyRings', resource_name='keyring', keyRingsId=KeyringAttributeConfig(kms_prefix), locationsId=LocationAttributeConfig(kms_prefix=kms_prefix, region_fallthrough=region_fallthrough), projectsId=ProjectAttributeConfig(kms_prefix), disable_auto_completers=False)