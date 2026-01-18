from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import log
@base.Deprecate(is_removed=True, warning='This command is deprecated. See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.', error='This command has been removed. See https://cloud.google.com/domains/docs/deprecations/feature-deprecations.')
class GetTransferParameters(base.DescribeCommand):
    """Get transfer parameters of a specific domain.

  Get parameters needed to transfer an existing domain from a different
  registrar. The parameters include the current registrar, name servers,
  transfer lock state, price, and supported privacy modes.

  ## EXAMPLES

  To check if ``example.com'' is available for transfer, run:

    $ {command} example.com
  """

    @staticmethod
    def Args(parser):
        resource_args.AddLocationResourceArg(parser)
        base.Argument('domain', help='Domain to get transfer parameters for.').AddToParser(parser)

    def Run(self, args):
        """Run the get transfer parameters command."""
        api_version = registrations.GetApiVersionFromArgs(args)
        client = registrations.RegistrationsClient(api_version)
        location_ref = args.CONCEPTS.location.Parse()
        domain = util.NormalizeDomainName(args.domain)
        if domain != args.domain:
            log.status.Print("Domain name '{}' has been normalized to equivalent '{}'.".format(args.domain, domain))
        return client.RetrieveTransferParameters(location_ref, domain)