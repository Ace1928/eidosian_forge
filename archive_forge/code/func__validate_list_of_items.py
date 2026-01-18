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
def _validate_list_of_items(item_validator, data, *args, **kwargs):
    if not isinstance(data, list):
        msg = _("'%s' is not a list") % data
        return msg
    dups = _collect_duplicates(data)
    if dups:
        msg = _("Duplicate items in the list: '%s'") % ', '.join([str(d) for d in dups])
        return msg
    for item in data:
        msg = item_validator(item, *args, **kwargs)
        if msg:
            return msg