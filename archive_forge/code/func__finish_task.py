import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
def _finish_task(self, task):
    try:
        task.succeed({'image_id': self.action_wrapper.image_id})
    except Exception as e:
        log_msg = _LE('Task ID %(task_id)s failed. Error: %(exc_type)s: %(e)s')
        LOG.exception(log_msg, {'exc_type': str(type(e)), 'e': encodeutils.exception_to_unicode(e), 'task_id': task.task_id})
        err_msg = _('Error: %(exc_type)s: %(e)s')
        task.fail(err_msg % {'exc_type': str(type(e)), 'e': encodeutils.exception_to_unicode(e)})
    finally:
        self.task_repo.save(task)