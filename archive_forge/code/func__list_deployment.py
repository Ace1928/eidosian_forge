import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _list_deployment(heat_client, args=None):
    kwargs = {'server_id': args.server} if args.server else {}
    columns = ['id', 'config_id', 'server_id', 'action', 'status']
    if args.long:
        columns.append('creation_time')
        columns.append('status_reason')
    deployments = heat_client.software_deployments.list(**kwargs)
    return (columns, (utils.get_item_properties(s, columns) for s in deployments))