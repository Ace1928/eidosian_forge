import datetime
import logging
from keystoneclient import exceptions as identity_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteTrust(command.Command):
    _description = _('Delete trust(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteTrust, self).get_parser(prog_name)
        parser.add_argument('trust', metavar='<trust>', nargs='+', help=_('Trust(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for trust in parsed_args.trust:
            try:
                trust_obj = utils.find_resource(identity_client.trusts, trust)
                identity_client.trusts.delete(trust_obj.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete trust with name or ID '%(trust)s': %(e)s"), {'trust': trust, 'e': e})
        if errors > 0:
            total = len(parsed_args.trust)
            msg = _('%(errors)s of %(total)s trusts failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)