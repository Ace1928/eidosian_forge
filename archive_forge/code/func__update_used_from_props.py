import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _update_used_from_props(columns, props):
    index_used = columns.index('used')
    updated_used = _exchange_dict_keys_with_values(props[index_used])
    for k, v in updated_used.items():
        updated_used[k] = list(_get_ranges(v))
    props = _hack_tuple_value_update_by_index(props, index_used, updated_used)
    return props