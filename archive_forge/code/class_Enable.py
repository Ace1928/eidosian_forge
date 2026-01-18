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
class Enable(base.SilentCommand):
    """Enable a root certificate authority.

    Enables a root certificate authority. The root certificate authority will be
    allowed to issue certificates once enabled.

    ## EXAMPLES

    To enable a root CA:

        $ {command} prod-root --location=us-west1 --pool=my-pool
  """

    @staticmethod
    def Args(parser):
        resource_args.AddCertAuthorityPositionalResourceArg(parser, 'to enable')

    def Run(self, args):
        client = privateca_base.GetClientInstance(api_version='v1')
        messages = privateca_base.GetMessagesModule(api_version='v1')
        ca_ref = args.CONCEPTS.certificate_authority.Parse()
        current_ca = client.projects_locations_caPools_certificateAuthorities.Get(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest(name=ca_ref.RelativeName()))
        resource_args.CheckExpectedCAType(messages.CertificateAuthority.TypeValueValuesEnum.SELF_SIGNED, current_ca, version='v1')
        operation = client.projects_locations_caPools_certificateAuthorities.Enable(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest(name=ca_ref.RelativeName(), enableCertificateAuthorityRequest=messages.EnableCertificateAuthorityRequest(requestId=request_utils.GenerateRequestId())))
        operations.Await(operation, 'Enabling Root CA', api_version='v1')
        log.status.Print('Enabled Root CA [{}].'.format(ca_ref.RelativeName()))