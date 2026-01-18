import uuid
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _get_migration_by_uuid(compute_client, server_id, migration_uuid):
    for migration in compute_client.server_migrations(server_id):
        if migration.uuid == migration_uuid:
            return migration
            break
    else:
        msg = _('In-progress live migration %s is not found for server %s.')
        raise exceptions.CommandError(msg % (migration_uuid, server_id))