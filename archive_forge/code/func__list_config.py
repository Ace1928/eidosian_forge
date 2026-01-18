import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import template_format
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
def _list_config(heat_client, args):
    kwargs = {}
    if args.limit:
        kwargs['limit'] = args.limit
    if args.marker:
        kwargs['marker'] = args.marker
    scs = heat_client.software_configs.list(**kwargs)
    columns = ['id', 'name', 'group', 'creation_time']
    return (columns, (utils.get_item_properties(s, columns) for s in scs))