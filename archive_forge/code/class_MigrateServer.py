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
class MigrateServer(command.Command):
    _description = _('Migrate server to different host.\n\nA migrate operation is implemented as a resize operation using the same flavor\nas the old server. This means that, like resize, migrate works by creating a\nnew server using the same flavor and copying the contents of the original disk\ninto a new one. As with resize, the migrate operation is a two-step process for\nthe user: the first step is to perform the migrate, and the second step is to\neither confirm (verify) success and release the old server, or to declare a\nrevert to release the new server and restart the old one.')

    def get_parser(self, prog_name):
        parser = super(MigrateServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('--live-migration', dest='live_migration', action='store_true', help=_('Live migrate the server; use the ``--host`` option to specify a target host for the migration which will be validated by the scheduler'))
        parser.add_argument('--host', metavar='<hostname>', help=_('Migrate the server to the specified host. (supported with --os-compute-api-version 2.30 or above when used with the --live-migration option) (supported with --os-compute-api-version 2.56 or above when used without the --live-migration option)'))
        migration_group = parser.add_mutually_exclusive_group()
        migration_group.add_argument('--shared-migration', dest='block_migration', action='store_false', default=None, help=_('Perform a shared live migration (default before --os-compute-api-version 2.25, auto after)'))
        migration_group.add_argument('--block-migration', dest='block_migration', action='store_true', help=_('Perform a block live migration (auto-configured from --os-compute-api-version 2.25)'))
        disk_group = parser.add_mutually_exclusive_group()
        disk_group.add_argument('--disk-overcommit', action='store_true', default=None, help=_('Allow disk over-commit on the destination host(supported with --os-compute-api-version 2.24 or below)'))
        disk_group.add_argument('--no-disk-overcommit', dest='disk_overcommit', action='store_false', help=_('Do not over-commit disk on the destination host (default)(supported with --os-compute-api-version 2.24 or below)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for migrate to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.compute
        server = utils.find_resource(compute_client.servers, parsed_args.server)
        if parsed_args.live_migration:
            kwargs = {}
            block_migration = parsed_args.block_migration
            if block_migration is None:
                if compute_client.api_version < api_versions.APIVersion('2.25'):
                    block_migration = False
                else:
                    block_migration = 'auto'
            kwargs['block_migration'] = block_migration
            if parsed_args.host and compute_client.api_version < api_versions.APIVersion('2.30'):
                raise exceptions.CommandError('--os-compute-api-version 2.30 or greater is required when using --host')
            kwargs['host'] = parsed_args.host
            if compute_client.api_version < api_versions.APIVersion('2.25'):
                kwargs['disk_over_commit'] = parsed_args.disk_overcommit
                if kwargs['disk_over_commit'] is None:
                    kwargs['disk_over_commit'] = False
            elif parsed_args.disk_overcommit is not None:
                msg = _('The --disk-overcommit and --no-disk-overcommit options are only supported by --os-compute-api-version 2.24 or below; this will be an error in a future release')
                self.log.warning(msg)
            server.live_migrate(**kwargs)
        else:
            if parsed_args.block_migration or parsed_args.disk_overcommit:
                raise exceptions.CommandError('--live-migration must be specified if --block-migration or --disk-overcommit is specified')
            if parsed_args.host:
                if compute_client.api_version < api_versions.APIVersion('2.56'):
                    msg = _('--os-compute-api-version 2.56 or greater is required to use --host without --live-migration.')
                    raise exceptions.CommandError(msg)
            kwargs = {'host': parsed_args.host} if parsed_args.host else {}
            server.migrate(**kwargs)
        if parsed_args.wait:
            if utils.wait_for_status(compute_client.servers.get, server.id, success_status=['active', 'verify_resize'], callback=_show_progress):
                self.app.stdout.write(_('Complete\n'))
            else:
                msg = _('Error migrating server: %s') % server.id
                raise exceptions.CommandError(msg)