import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteDomain(command.Command):
    _description = _('Delete domain(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteDomain, self).get_parser(prog_name)
        parser.add_argument('domain', metavar='<domain>', nargs='+', help=_('Domain(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.domain:
            try:
                domain = utils.find_resource(identity_client.domains, i)
                identity_client.domains.delete(domain.id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete domain with name or ID '%(domain)s': %(e)s"), {'domain': i, 'e': e})
        if result > 0:
            total = len(parsed_args.domain)
            msg = _('%(result)s of %(total)s domains failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)