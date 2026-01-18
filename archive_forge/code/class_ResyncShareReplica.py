import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ResyncShareReplica(command.Command):
    """Resync share replica"""
    _description = _("Attempt to update the share replica with its 'active' mirror.")

    def get_parser(self, prog_name):
        parser = super(ResyncShareReplica, self).get_parser(prog_name)
        parser.add_argument('replica', metavar='<replica>', help=_('ID of the share replica to resync.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        replica = osc_utils.find_resource(share_client.share_replicas, parsed_args.replica)
        try:
            share_client.share_replicas.resync(replica)
        except Exception as e:
            raise exceptions.CommandError(_('Failed to resync share replica: %s' % e))