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
def _delete_encryption_key(self, context, image):
    props = image.extra_properties
    cinder_encryption_key_id = props.get('cinder_encryption_key_id')
    if cinder_encryption_key_id is None:
        return
    deletion_policy = props.get('cinder_encryption_key_deletion_policy', '')
    if deletion_policy != 'on_image_deletion':
        return
    try:
        self._key_manager.delete(context, cinder_encryption_key_id)
    except castellan_exception.Forbidden:
        msg = 'Not allowed to delete encryption key %s' % cinder_encryption_key_id
        LOG.warning(msg)
    except (castellan_exception.ManagedObjectNotFoundError, KeyError):
        msg = 'Could not find encryption key %s' % cinder_encryption_key_id
        LOG.warning(msg)
    except castellan_exception.KeyManagerError:
        msg = 'Failed to delete cinder encryption key %s' % cinder_encryption_key_id
        LOG.warning(msg)