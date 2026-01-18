import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def _get_store_id_from_uri(uri):
    scheme = urlparse.urlparse(uri).scheme
    location_map = store_api.location.SCHEME_TO_CLS_BACKEND_MAP
    url_matched = False
    if scheme not in location_map:
        LOG.warning("Unknown scheme '%(scheme)s' found in uri '%(uri)s'", {'scheme': scheme, 'uri': uri})
        return
    for store in location_map[scheme]:
        store_instance = location_map[scheme][store]['store']
        url_prefix = store_instance.url_prefix
        if url_prefix and uri.startswith(url_prefix):
            url_matched = True
            break
    if url_matched:
        return u'%s' % store
    else:
        LOG.warning('Invalid location uri %s', uri)
        return