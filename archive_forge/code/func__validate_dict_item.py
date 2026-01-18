import collections
import functools
import inspect
import re
import netaddr
from os_ken.lib.packet import ether_types
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from webob import exc
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.plugins import directory
from neutron_lib.services.qos import constants as qos_consts
def _validate_dict_item(key, key_validator, data):
    conv_func = key_validator.get('convert_to')
    if conv_func:
        data[key] = conv_func(data.get(key))
    try:
        dummy_, val_func, val_params = _extract_validator(key_validator)
        if val_func:
            return val_func(data.get(key), val_params)
    except UndefinedValidator as e:
        LOG.debug(e.message)
        return e.message