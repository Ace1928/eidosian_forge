import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def _list_driver_opts():
    driver_opts = {}
    mgr = extension.ExtensionManager('glance_store.drivers')
    drivers = sorted([ext.name for ext in mgr])
    handled_drivers = []
    for store_entry in drivers:
        driver_cls = _load_multi_store(None, store_entry, False)
        if driver_cls and driver_cls not in handled_drivers:
            if getattr(driver_cls, 'OPTIONS', None) is not None:
                driver_opts[store_entry] = driver_cls.OPTIONS
            handled_drivers.append(driver_cls)
    return driver_opts