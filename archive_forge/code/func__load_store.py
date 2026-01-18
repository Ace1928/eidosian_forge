import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def _load_store(conf, store_entry, invoke_load=True):
    try:
        LOG.debug('Attempting to import store %s', store_entry)
        mgr = driver.DriverManager('glance_store.drivers', store_entry, invoke_args=[conf], invoke_on_load=invoke_load)
        return mgr.driver
    except RuntimeError as e:
        LOG.warning('Failed to load driver %(driver)s. The driver will be disabled' % dict(driver=str([driver, e])))