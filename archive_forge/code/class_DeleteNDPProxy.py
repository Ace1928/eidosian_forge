import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteNDPProxy(command.Command):
    _description = _('Delete NDP proxy')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('ndp_proxy', nargs='+', metavar='<ndp-proxy>', help=_('NDP proxy(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for ndp_proxy in parsed_args.ndp_proxy:
            try:
                obj = client.find_ndp_proxy(ndp_proxy, ignore_missing=False)
                client.delete_ndp_proxy(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete NDP proxy '%(ndp_proxy)s': %(e)s"), {'ndp_proxy': ndp_proxy, 'e': e})
        if result > 0:
            total = len(parsed_args.ndp_proxy)
            msg = _('%(result)s of %(total)s NDP proxies failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)