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
def _cleanup_stale_task_progress(self, image_repo, image, task):
    """Cleanup stale in-progress information from a previous task.

        If we stole the lock from another task, we should try to clean up
        the in-progress status information from that task while we have
        the lock.
        """
    stores = task.task_input.get('backend', [])
    keys = ['os_glance_importing_to_stores', 'os_glance_failed_import']
    changed = set()
    for store in stores:
        for key in keys:
            values = image.extra_properties.get(key, '').split(',')
            if store in values:
                values.remove(store)
                changed.add(key)
            image.extra_properties[key] = ','.join(values)
    if changed:
        image_repo.save(image)
        LOG.debug('Image %(image)s had stale import progress info %(keys)s from task %(task)s which was cleaned up', {'image': image.image_id, 'task': task.task_id, 'keys': ','.join(changed)})