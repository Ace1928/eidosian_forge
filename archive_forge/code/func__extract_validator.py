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
def _extract_validator(key_validator):
    for k, v in key_validator.items():
        if k.startswith('type:'):
            validator_name, validator_params = (k, v)
            try:
                return (validator_name, validators[validator_name], validator_params)
            except KeyError as e:
                raise UndefinedValidator(validator_name) from e
    return (None, None, None)