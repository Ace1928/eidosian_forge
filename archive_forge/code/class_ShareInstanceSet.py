import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
class ShareInstanceSet(command.Command):
    """Set share instance"""
    _description = _('Explicitly reset share instance status')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceSet, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('Instance to be modified.'))
        parser.add_argument('--status', metavar='<status>', help=_('Indicate which state to assign the instance. Options are: available, error, creating, deleting,error_deleting, migrating, migrating_to, server_migrating.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        instance = osc_utils.find_resource(share_client.share_instances, parsed_args.instance)
        if parsed_args.status:
            try:
                share_client.share_instances.reset_state(instance, parsed_args.status)
            except Exception as e:
                LOG.error(_("Failed to set status '%(status)s': %(exception)s"), {'status': parsed_args.status, 'exception': e})
                raise exceptions.CommandError(_('Set operation failed'))
        if not instance or not parsed_args.status:
            raise exceptions.CommandError(_("Nothing to set. Please define a '--status'."))