import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def _validate_validation_data(self, image, locations):
    val_data = {}
    for loc in locations:
        if 'validation_data' not in loc:
            continue
        for k, v in loc['validation_data'].items():
            if val_data.get(k, v) != v:
                msg = _('Conflicting values for %s') % k
                raise webob.exc.HTTPConflict(explanation=msg)
            val_data[k] = v
    new_val_data = {}
    for k, v in val_data.items():
        current = getattr(image, k)
        if v == current:
            continue
        if current:
            msg = _('%s is already set with a different value') % k
            raise webob.exc.HTTPConflict(explanation=msg)
        new_val_data[k] = v
    if not new_val_data:
        return {}
    if image.status != 'queued':
        msg = _("New value(s) for %s may only be provided when image status is 'queued'") % ', '.join(new_val_data.keys())
        raise webob.exc.HTTPConflict(explanation=msg)
    if 'checksum' in new_val_data:
        try:
            checksum_bytes = bytearray.fromhex(new_val_data['checksum'])
        except ValueError:
            msg = _('checksum (%s) is not a valid hexadecimal value') % new_val_data['checksum']
            raise webob.exc.HTTPConflict(explanation=msg)
        if len(checksum_bytes) != 16:
            msg = _('checksum (%s) is not the correct size for md5 (should be 16 bytes)') % new_val_data['checksum']
            raise webob.exc.HTTPConflict(explanation=msg)
    hash_algo = new_val_data.get('os_hash_algo')
    if hash_algo != CONF['hashing_algorithm']:
        msg = _('os_hash_algo must be %(want)s, not %(got)s') % {'want': CONF['hashing_algorithm'], 'got': hash_algo}
        raise webob.exc.HTTPConflict(explanation=msg)
    try:
        hash_bytes = bytearray.fromhex(new_val_data['os_hash_value'])
    except ValueError:
        msg = _('os_hash_value (%s) is not a valid hexadecimal value') % new_val_data['os_hash_value']
        raise webob.exc.HTTPConflict(explanation=msg)
    want_size = hashlib.new(hash_algo).digest_size
    if len(hash_bytes) != want_size:
        msg = _('os_hash_value (%(value)s) is not the correct size for %(algo)s (should be %(want)d bytes)') % {'value': new_val_data['os_hash_value'], 'algo': hash_algo, 'want': want_size}
        raise webob.exc.HTTPConflict(explanation=msg)
    return new_val_data