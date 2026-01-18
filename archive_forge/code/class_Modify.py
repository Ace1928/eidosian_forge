from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.commerce_procurement import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.commerce_procurement import resource_args
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Modify(base.Command):
    """Modifies the order resource from the Modify API."""

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: argparse.ArgumentParser to register arguments with.
    """
        resource_args.AddOrderResourceArg(parser, 'Order to modify.')
        parser.add_argument('--etag', help='The weak etag for validation check, if specified.')
        product_quote_group = parser.add_mutually_exclusive_group(required=True)
        product_quote_group.add_argument('--product-request', type=arg_parsers.ArgDict(required_keys=['line-item-id', 'line-item-change-type']), metavar='KEY=VALUE', action='append', help='Request about product info to modify order against.')
        quote_request_group = product_quote_group.add_group(help='Quote-related modification.')
        quote_request_group.add_argument('--quote-change-type', required=True, help='Change type on quote based purchase.')
        quote_request_group.add_argument('--new-quote-external-name', help='The external name of the quote the order will use.')

    def Run(self, args):
        """Runs the command.

    Args:
      args: The arguments that were provided to this command invocation.

    Returns:
      An Order operation.
    """
        order_ref = args.CONCEPTS.order.Parse()
        return apis.Orders.Modify(order_ref.RelativeName(), args.etag, args.product_request, args.quote_change_type, args.new_quote_external_name)