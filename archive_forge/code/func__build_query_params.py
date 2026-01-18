import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
@staticmethod
def _build_query_params(marker=None, limit=None, sort_keys=None, sort_dirs=None, fields=None, filters=None, scope=None, namespace=None):
    qparams = {}
    if marker:
        qparams['marker'] = marker
    if limit and limit > 0:
        qparams['limit'] = limit
    if sort_keys:
        qparams['sort_keys'] = sort_keys
    if sort_dirs:
        qparams['sort_dirs'] = sort_dirs
    if fields:
        qparams['fields'] = ','.join(fields)
    if filters:
        for name, val in filters.items():
            qparams[name] = val
    if scope:
        qparams['scope'] = scope
    if namespace:
        qparams['namespace'] = namespace
    return '?%s' % urlparse.urlencode(list(qparams.items())) if qparams else ''