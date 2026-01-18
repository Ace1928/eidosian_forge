from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Undelete(base.SilentCommand):
    """Undelete a root Certificate Authority.

    Restores a root Certificate Authority that has been deleted. A Certificate
    Authority can be undeleted within 30 days of being deleted. Use this command
    to halt the deletion process. An undeleted CA will move to DISABLED state.

    ## EXAMPLES

    To undelete a root CA:

        $ {command} prod-root --location=us-west1 --pool=my-pool
  """

    @staticmethod
    def Args(parser):
        resource_args.AddCertAuthorityPositionalResourceArg(parser, 'to undelete')

    def Run(self, args):
        client = privateca_base.GetClientInstance(api_version='v1')
        messages = privateca_base.GetMessagesModule(api_version='v1')
        ca_ref = args.CONCEPTS.certificate_authority.Parse()
        current_ca = client.projects_locations_caPools_certificateAuthorities.Get(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest(name=ca_ref.RelativeName()))
        resource_args.CheckExpectedCAType(messages.CertificateAuthority.TypeValueValuesEnum.SELF_SIGNED, current_ca, version='v1')
        operation = client.projects_locations_caPools_certificateAuthorities.Undelete(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest(name=ca_ref.RelativeName(), undeleteCertificateAuthorityRequest=messages.UndeleteCertificateAuthorityRequest(requestId=request_utils.GenerateRequestId())))
        operations.Await(operation, 'Undeleting Root CA', api_version='v1')
        log.status.Print('Undeleted Root CA [{}].'.format(ca_ref.RelativeName()))