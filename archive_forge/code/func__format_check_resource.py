import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _format_check_resource(item):
    item_id = getattr(item, 'id', False)
    if item_id == 'dry-run=pass':
        item.check_resource = 'pass'
    return item