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
def _validate_any_key_spec(data, key_specs=None):
    """Validate a dict matches at least 1 key spec.

    :param data: The dict to validate.
    :param key_specs: An iterable collection of key spec dicts used to validate
        data.
    :returns: None if at least 1 of the key_specs matches data, otherwise
        a message is returned indicating data could not be matched with any
        of the key_specs.
    """
    for spec in key_specs:
        if validate_dict(data, spec) is None:
            return None
    msg = 'No valid key specs matched for: %s'
    LOG.debug(msg, data)
    return _(msg) % data