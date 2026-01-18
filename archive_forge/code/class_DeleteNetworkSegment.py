import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class DeleteNetworkSegment(command.Command):
    _description = _('Delete network segment(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkSegment, self).get_parser(prog_name)
        parser.add_argument('network_segment', metavar='<network-segment>', nargs='+', help=_('Network segment(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for network_segment in parsed_args.network_segment:
            try:
                obj = client.find_segment(network_segment, ignore_missing=False)
                client.delete_segment(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete network segment with ID '%(network_segment)s': %(e)s"), {'network_segment': network_segment, 'e': e})
        if result > 0:
            total = len(parsed_args.network_segment)
            msg = _('%(result)s of %(total)s network segments failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)