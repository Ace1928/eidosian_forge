import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def _new_location(self, old_location, url):
    store_name = old_location.store_name
    store_class = old_location.store_location.__class__
    image_id = old_location.image_id
    store_specs = old_location.store_specs
    parsed_url = urllib.parse.urlparse(url)
    new_url = parsed_url._replace(scheme='vsphere')
    vsphere_url = urllib.parse.urlunparse(new_url)
    return glance_store.location.Location(store_name, store_class, self.conf, uri=vsphere_url, image_id=image_id, store_specs=store_specs, backend=self.backend_group)