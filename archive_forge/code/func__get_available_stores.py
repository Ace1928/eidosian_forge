import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def _get_available_stores():
    available_stores = CONF.enabled_backends
    stores = []
    for store in available_stores:
        if available_stores[store] == 'http':
            continue
        if store not in wsgi.RESERVED_STORES:
            stores.append(store)
    return stores