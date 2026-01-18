import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
def _list_snapshot(self, heat_client, parsed_args):
    fields = {'stack_id': parsed_args.stack}
    try:
        snapshots = heat_client.stacks.snapshot_list(**fields)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % parsed_args.stack)
    columns = ['id', 'name', 'status', 'status_reason', 'creation_time']
    return (columns, (utils.get_dict_properties(s, columns) for s in snapshots['snapshots']))