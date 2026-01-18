from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.kms_configs import client as kmsconfigs_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Verify(base.Command):
    """Verify that the Cloud NetApp Volumes KMS Config is reachable."""
    detailed_help = {'DESCRIPTION': '          Verifies that the Cloud NetApp Volumes KMS (Key Management System) Config is reachable.\n          ', 'EXAMPLES': '          The following command verifies that the KMS Config instance named KMS_CONFIG is reachable using specified location.\n\n              $ {command} KMS_CONFIG --location=us-central1\n          '}
    _RELEASE_TRACK = base.ReleaseTrack.GA

    @staticmethod
    def Args(parser):
        concept_parsers.ConceptParser([flags.GetKmsConfigPresentationSpec('The KMS Config used to verify')]).AddToParser(parser)

    def Run(self, args):
        """Verify that the Cloud NetApp Volumes KMS Config is reachable."""
        kmsconfig_ref = args.CONCEPTS.kms_config.Parse()
        client = kmsconfigs_client.KmsConfigsClient(self._RELEASE_TRACK)
        return client.VerifyKmsConfig(kmsconfig_ref)