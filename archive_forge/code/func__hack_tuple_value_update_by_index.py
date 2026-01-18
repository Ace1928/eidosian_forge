import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _hack_tuple_value_update_by_index(tup, index, value):
    lot = list(tup)
    lot[index] = value
    return tuple(lot)