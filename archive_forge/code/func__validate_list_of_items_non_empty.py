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
def _validate_list_of_items_non_empty(item_validator, data, *args, **kwargs):
    res = _validate_list_of_items(item_validator, data, *args, **kwargs)
    if res is not None:
        return res
    if len(data) == 0:
        msg = _('List should not be empty')
        return msg