import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _update_available_from_props(columns, props):
    index_available = columns.index('available')
    props = _hack_tuple_value_update_by_index(props, index_available, list(_get_ranges(props[index_available])))
    return props