from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetKmsKeyVersionResourceSpec(kms_prefix=True):
    return concepts.ResourceSpec('cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions', resource_name='key version', cryptoKeyVersionsId=KeyVersionAttributeConfig(kms_prefix), cryptoKeysId=KeyAttributeConfig(kms_prefix), keyRingsId=KeyringAttributeConfig(kms_prefix), locationsId=LocationAttributeConfig(kms_prefix=kms_prefix), projectsId=ProjectAttributeConfig(kms_prefix=kms_prefix), disable_auto_completers=False)