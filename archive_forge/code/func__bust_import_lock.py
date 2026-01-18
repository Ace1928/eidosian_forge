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
def _bust_import_lock(self, admin_image_repo, admin_task_repo, image, task, task_id):
    if task:
        try:
            task.fail('Expired lock preempted')
            admin_task_repo.save(task)
        except exception.InvalidTaskStatusTransition:
            pass
    try:
        admin_image_repo.delete_property_atomic(image, 'os_glance_import_task', task_id)
    except exception.NotFound:
        LOG.warning('Image %(image)s has stale import task %(task)s but we lost the race to remove it.', {'image': image.image_id, 'task': task_id})
        raise exception.Conflict('Image has active task')
    LOG.warning('Image %(image)s has stale import task %(task)s in status %(status)s from %(owner)s; removed lock because it had expired.', {'image': image.image_id, 'task': task_id, 'status': task and task.status or 'missing', 'owner': task and task.owner or 'unknown owner'})