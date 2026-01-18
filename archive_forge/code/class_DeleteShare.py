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
class DeleteShare(command.Command):
    """Delete a share."""
    _description = _('Delete a share')

    def get_parser(self, prog_name):
        parser = super(DeleteShare, self).get_parser(prog_name)
        parser.add_argument('shares', metavar='<share>', nargs='+', help=_('Share(s) to delete (name or ID)'))
        parser.add_argument('--share-group', metavar='<share-group>', default=None, help=_('Optional share group (name or ID) which contains the share'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Attempt forced removal of share(s), regardless of state (defaults to False)'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share deletion'))
        parser.add_argument('--soft', action='store_true', default=False, help=_('Soft delete one or more shares.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share in parsed_args.shares:
            try:
                share_obj = apiutils.find_resource(share_client.shares, share)
                share_group_id = None
                if parsed_args.share_group:
                    share_group_id = apiutils.find_resource(share_client.share_groups, parsed_args.share_group).id
                if parsed_args.force:
                    share_client.shares.force_delete(share_obj)
                elif parsed_args.soft:
                    if share_client.api_version >= api_versions.APIVersion('2.69'):
                        share_client.shares.soft_delete(share_obj)
                    else:
                        raise exceptions.CommandError('Soft Deleting shares is only available with manila API version >= 2.69')
                else:
                    share_client.shares.delete(share_obj, share_group_id)
                if parsed_args.wait:
                    if not oscutils.wait_for_delete(manager=share_client.shares, res_id=share_obj.id):
                        result += 1
            except Exception as exc:
                result += 1
                LOG.error(_("Failed to delete share with name or ID '%(share)s': %(e)s"), {'share': share, 'e': exc})
        if result > 0:
            total = len(parsed_args.shares)
            msg = _('%(result)s of %(total)s shares failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)