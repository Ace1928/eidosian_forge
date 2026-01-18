import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _get_item_properties(item, fields):
    """Return a tuple containing the item properties."""
    row = []
    for field in fields:
        row.append(item.get(field, ''))
    return tuple(row)