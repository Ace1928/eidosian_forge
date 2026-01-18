import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _exchange_dict_keys_with_values(orig_dict):
    updated_dict = dict()
    for k, v in orig_dict.items():
        k = [k]
        if not updated_dict.get(v):
            updated_dict[v] = k
        else:
            updated_dict[v].extend(k)
    return updated_dict