import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteLocalIP(command.Command):
    _description = _('Delete local IP(s)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', nargs='+', help=_('Local IP(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for lip in parsed_args.local_ip:
            try:
                obj = client.find_local_ip(lip, ignore_missing=False)
                client.delete_local_ip(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete Local IP with name or ID '%(lip)s': %(e)s"), {'lip': lip, 'e': e})
        if result > 0:
            total = len(parsed_args.local_ip)
            msg = _('%(result)s of %(total)s local IPs failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)