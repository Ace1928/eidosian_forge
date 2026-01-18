import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _update_additional_fields_from_props(columns, props):
    props = _update_available_from_props(columns, props)
    props = _update_used_from_props(columns, props)
    return props