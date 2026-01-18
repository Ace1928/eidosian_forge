import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class AbandonShare(command.Command):
    """Abandon a share (Admin only)."""
    _description = _('Abandon a share')

    def get_parser(self, prog_name):
        parser = super(AbandonShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', nargs='+', help=_('Name or ID of the share(s)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until share is abandoned'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share in parsed_args.share:
            try:
                share_obj = apiutils.find_resource(share_client.shares, share)
                share_client.shares.unmanage(share_obj)
                if parsed_args.wait:
                    if not oscutils.wait_for_delete(manager=share_client.shares, res_id=share_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to abandon share with name or ID '%(share)s': %(e)s"), {'share': share, 'e': e})
        if result > 0:
            total = len(parsed_args.share)
            msg = _('Failed to abandon %(result)s out of %(total)s shares.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)