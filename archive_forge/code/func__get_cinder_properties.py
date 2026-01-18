import copy
import glance_store as g_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_serialization.jsonutils as json
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
from glance.i18n import _
from glance.quota import keystone as ks_quota
@staticmethod
def _get_cinder_properties(store_detail):
    return {'volume_type': store_detail.store_conf.cinder_volume_type, 'use_multipath': store_detail.store_conf.cinder_use_multipath}