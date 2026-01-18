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
def _check_image_location(context, store_api, store_utils, location):
    backend = None
    if CONF.enabled_backends:
        backend = location['metadata'].get('store')
    _check_location_uri(context, store_api, store_utils, location['url'], backend=backend)
    store_api.check_location_metadata(location['metadata'])