import urllib.parse as urlparse
from oslo_config import cfg
from glance.i18n import _
def _foreach_store_type_preference():
    store_types = CONF.store_type_location_strategy.store_type_preference
    for preferred_store in store_types:
        preferred_store = str(preferred_store).strip()
        if not preferred_store:
            continue
        yield preferred_store