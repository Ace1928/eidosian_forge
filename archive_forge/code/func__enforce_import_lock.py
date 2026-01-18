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
def _enforce_import_lock(self, req, image):
    admin_context = req.context.elevated()
    admin_image_repo = self.gateway.get_repo(admin_context)
    admin_task_repo = self.gateway.get_task_repo(admin_context)
    other_task = image.extra_properties['os_glance_import_task']
    expiry = datetime.timedelta(minutes=60)
    bustable_states = ('pending', 'processing', 'success', 'failure')
    try:
        task = admin_task_repo.get(other_task)
    except exception.NotFound:
        LOG.warning('Image %(image)s has non-existent import task %(task)s; considering it stale', {'image': image.image_id, 'task': other_task})
        task = None
        age = 0
    else:
        age = oslo_timeutils.utcnow() - task.updated_at
        if task.status == 'pending':
            expiry *= 2
    if not task or (task.status in bustable_states and age >= expiry):
        self._bust_import_lock(admin_image_repo, admin_task_repo, image, task, other_task)
        return task
    if task.status in bustable_states:
        LOG.warning('Image %(image)s has active import task %(task)s in status %(status)s; lock remains valid for %(expire)i more seconds', {'image': image.image_id, 'task': task.task_id, 'status': task.status, 'expire': (expiry - age).total_seconds()})
    else:
        LOG.debug('Image %(image)s has import task %(task)s in status %(status)s and does not qualify for expiry.', {'image': image.image_id, 'task': task.task_id, 'status': task.status})
    raise exception.Conflict('Image has active task')