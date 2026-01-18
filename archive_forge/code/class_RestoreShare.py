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
class RestoreShare(command.Command):
    """Restore one or more shares from recycle bin"""
    _description = _('Restores this share or more shares from the recycle bin')

    def get_parser(self, prog_name):
        parser = super(RestoreShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', nargs='+', help=_('Name or ID of the share(s)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if share_client.api_version >= api_versions.APIVersion('2.69'):
            failure_count = 0
            for share in parsed_args.share:
                try:
                    share_client.shares.restore(share)
                except Exception as e:
                    failure_count += 1
                    LOG.error(_("Failed to restore share with name or ID '%(share)s': %(e)s"), {'share': share, 'e': e})
            if failure_count > 0:
                total = len(parsed_args.share)
                msg = f'Failed to restore {failure_count} out of {total} shares.'
                msg = _(msg)
                raise exceptions.CommandError(msg)
        else:
            raise exceptions.CommandError('Restoring a share from the recycle bin is only available with manila API version >= 2.69')