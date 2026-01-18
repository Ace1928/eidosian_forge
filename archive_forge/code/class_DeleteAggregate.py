import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteAggregate(command.Command):
    _description = _('Delete existing aggregate(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteAggregate, self).get_parser(prog_name)
        parser.add_argument('aggregate', metavar='<aggregate>', nargs='+', help=_('Aggregate(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        result = 0
        for a in parsed_args.aggregate:
            try:
                aggregate = compute_client.find_aggregate(a, ignore_missing=False)
                compute_client.delete_aggregate(aggregate.id, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete aggregate with name or ID '%(aggregate)s': %(e)s"), {'aggregate': a, 'e': e})
        if result > 0:
            total = len(parsed_args.aggregate)
            msg = _('%(result)s of %(total)s aggregates failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)