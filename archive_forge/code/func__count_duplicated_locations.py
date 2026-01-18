from collections import abc
import copy
import functools
from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from glance.common import exception
from glance.common import format_inspector
from glance.common import store_utils
from glance.common import utils
import glance.domain.proxy
from glance.i18n import _, _LE, _LI, _LW
def _count_duplicated_locations(locations, new):
    """
    To calculate the count of duplicated locations for new one.

    :param locations: The exiting image location set
    :param new: The new image location
    :returns: The count of duplicated locations
    """
    ret = 0
    for loc in locations:
        if loc['url'] == new['url'] and loc['metadata'] == new['metadata']:
            ret += 1
    return ret