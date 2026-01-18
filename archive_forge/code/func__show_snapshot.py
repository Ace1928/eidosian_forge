import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
def _show_snapshot(self, heat_client, stack_id, snapshot_id):
    try:
        data = heat_client.stacks.snapshot_show(stack_id, snapshot_id)
    except heat_exc.HTTPNotFound:
        raise exc.CommandError(_('Snapshot ID <%(snapshot_id)s> not found for stack <%(stack_id)s>') % {'snapshot_id': snapshot_id, 'stack_id': stack_id})
    rows = list(data.values())
    columns = list(data.keys())
    return (columns, rows)