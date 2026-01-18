from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetCertificateChain(base.DescribeCommand):
    """Get a certificate chain for a given version.

  Returns the PEM-format certificate chain for the specified key version.
  The optional flag `output-file` indicates the path to store the PEM. If not
  specified, the PEM will be printed to stdout.
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddKeyVersionResourceArgument(parser, 'from which to get the certificate chain')
        flags.AddCertificateChainFlag(parser)
        flags.AddOutputFileFlag(parser, 'to store PEM')

    def Run(self, args):
        client = cloudkms_base.GetClientInstance()
        messages = cloudkms_base.GetMessagesModule()
        version_ref = flags.ParseCryptoKeyVersionName(args)
        if not version_ref.Name():
            raise exceptions.InvalidArgumentException('version', 'version id must be non-empty.')
        versions = client.projects_locations_keyRings_cryptoKeys_cryptoKeyVersions
        version = versions.Get(messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest(name=version_ref.RelativeName()))
        if version.protectionLevel != messages.CryptoKeyVersion.ProtectionLevelValueValuesEnum.HSM:
            raise kms_exceptions.ArgumentError('Certificate chains are only available for HSM key versions.')
        if version.state == messages.CryptoKeyVersion.StateValueValuesEnum.PENDING_GENERATION:
            raise kms_exceptions.ArgumentError('Certificate chains are unavailable until the version is generated.')
        try:
            log.WriteToFileOrStdout(args.output_file if args.output_file else '-', _GetCertificateChainPem(version.attestation.certChains, args.certificate_chain_type), overwrite=True, binary=False)
        except files.Error as e:
            raise exceptions.BadFileException(e)