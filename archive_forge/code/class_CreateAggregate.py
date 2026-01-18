import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateAggregate(command.ShowOne):
    _description = _('Create a new aggregate')

    def get_parser(self, prog_name):
        parser = super(CreateAggregate, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('New aggregate name'))
        parser.add_argument('--zone', metavar='<availability-zone>', help=_('Availability zone name'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Property to add to this aggregate (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        attrs = {'name': parsed_args.name}
        if parsed_args.zone:
            attrs['availability_zone'] = parsed_args.zone
        aggregate = compute_client.create_aggregate(**attrs)
        if parsed_args.properties:
            aggregate = compute_client.set_aggregate_metadata(aggregate.id, parsed_args.properties)
        display_columns, columns = _get_aggregate_columns(aggregate)
        data = utils.get_item_properties(aggregate, columns, formatters=_aggregate_formatters)
        return (display_columns, data)