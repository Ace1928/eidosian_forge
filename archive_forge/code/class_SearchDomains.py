from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
class SearchDomains(base.DescribeCommand):
    """Search for available domains.

  Search for available domains relevant to a specified query.

  This command uses cached domain name availability information. Use the
  get-register-params command to get up-to-date availability information.

  ## EXAMPLES

  To search for domains for ``my-new-project'', run:

    $ {command} my-new-project

  To search for a specific domain, like ``example.com'', and get suggestions for
  other domain endings, run:

    $ {command} example.com
  """

    @staticmethod
    def Args(parser):
        resource_args.AddLocationResourceArg(parser, 'to search domains in')
        parser.display_info.AddTransforms({'price': util.TransformMoneyType})
        parser.display_info.AddFormat(_FORMAT)
        base.Argument('domain_query', help='Domain search query. May be a domain name or arbitrary search terms.').AddToParser(parser)

    def Run(self, args):
        """Run the search domains command."""
        api_version = registrations.GetApiVersionFromArgs(args)
        client = registrations.RegistrationsClient(api_version)
        location_ref = args.CONCEPTS.location.Parse()
        suggestions = client.SearchDomains(location_ref, args.domain_query)
        for s in suggestions:
            try:
                s.domainName = util.PunycodeToUnicode(s.domainName)
            except UnicodeError:
                pass
        if not suggestions:
            suggestions.append(client.messages.RegisterParameters())
        return suggestions