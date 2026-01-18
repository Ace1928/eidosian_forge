import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class ResizeServer(command.Command):
    _description = _('Scale server to a new flavor.\n\nA resize operation is implemented by creating a new server and copying the\ncontents of the original disk into a new one. It is a two-step process for the\nuser: the first step is to perform the resize, and the second step is to either\nconfirm (verify) success and release the old server or to declare a revert to\nrelease the new server and restart the old one.')

    def get_parser(self, prog_name):
        parser = super(ResizeServer, self).get_parser(prog_name)
        phase_group = parser.add_mutually_exclusive_group()
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        phase_group.add_argument('--flavor', metavar='<flavor>', help=_('Resize server to specified flavor'))
        phase_group.add_argument('--confirm', action='store_true', help=_("**Deprecated** Confirm server resize is complete. Replaced by the 'openstack server resize confirm' and 'openstack server migration confirm' commands"))
        phase_group.add_argument('--revert', action='store_true', help=_("**Deprecated** Restore server state before resizeReplaced by the 'openstack server resize revert' and 'openstack server migration revert' commands"))
        parser.add_argument('--wait', action='store_true', help=_('Wait for resize to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.compute
        server = utils.find_resource(compute_client.servers, parsed_args.server)
        if parsed_args.flavor:
            flavor = utils.find_resource(compute_client.flavors, parsed_args.flavor)
            if not server.image:
                self.log.warning(_('The root disk size in flavor will not be applied while booting from a persistent volume.'))
            compute_client.servers.resize(server, flavor)
            if parsed_args.wait:
                if utils.wait_for_status(compute_client.servers.get, server.id, success_status=['active', 'verify_resize'], callback=_show_progress):
                    self.app.stdout.write(_('Complete\n'))
                else:
                    msg = _('Error resizing server: %s') % server.id
                    raise exceptions.CommandError(msg)
        elif parsed_args.confirm:
            self.log.warning(_("The --confirm option has been deprecated. Please use the 'openstack server resize confirm' command instead."))
            compute_client.servers.confirm_resize(server)
        elif parsed_args.revert:
            self.log.warning(_("The --revert option has been deprecated. Please use the 'openstack server resize revert' command instead."))
            compute_client.servers.revert_resize(server)