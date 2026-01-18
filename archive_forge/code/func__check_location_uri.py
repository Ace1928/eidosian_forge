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
def _check_location_uri(context, store_api, store_utils, uri, backend=None):
    """Check if an image location is valid.

    :param context: Glance request context
    :param store_api: store API module
    :param store_utils: store utils module
    :param uri: location's uri string
    :param backend: A backend name for the store
    """
    try:
        if CONF.enabled_backends:
            size_from_backend = store_api.get_size_from_uri_and_backend(uri, backend, context=context)
        else:
            size_from_backend = store_api.get_size_from_backend(uri, context=context)
        is_ok = store_utils.validate_external_location(uri) and size_from_backend > 0
    except (store.UnknownScheme, store.NotFound, store.BadStoreUri):
        is_ok = False
    if not is_ok:
        reason = _('Invalid location')
        raise exception.BadStoreUri(message=reason)