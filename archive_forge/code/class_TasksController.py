import copy
import http.client as http
import urllib.parse as urlparse
import debtcollector
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_serialization.jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _, _LW
import glance.notifier
import glance.schema
class TasksController(object):
    """Manages operations on tasks."""

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None, store_api=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.store_api = store_api or glance_store
        self.gateway = glance.gateway.Gateway(self.db_api, self.store_api, self.notifier, self.policy)

    @debtcollector.removals.remove(message=_DEPRECATION_MESSAGE)
    def create(self, req, task):
        ctxt = req.context
        task_factory = self.gateway.get_task_factory(ctxt)
        executor_factory = self.gateway.get_task_executor_factory(ctxt)
        task_repo = self.gateway.get_task_repo(ctxt)
        try:
            new_task = task_factory.new_task(task_type=task['type'], owner=ctxt.owner, task_input=task['input'], image_id=task['input'].get('image_id'), user_id=ctxt.user_id, request_id=ctxt.request_id)
            task_repo.add(new_task)
            task_executor = executor_factory.new_task_executor(ctxt)
            pool = common.get_thread_pool('tasks_pool')
            pool.spawn(new_task.run, task_executor)
        except exception.Forbidden as e:
            msg = _LW('Forbidden to create task. Reason: %(reason)s') % {'reason': encodeutils.exception_to_unicode(e)}
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        return new_task

    @debtcollector.removals.remove(message=_DEPRECATION_MESSAGE)
    def index(self, req, marker=None, limit=None, sort_key='created_at', sort_dir='desc', filters=None):
        result = {}
        if filters is None:
            filters = {}
        filters['deleted'] = False
        if limit is None:
            limit = CONF.limit_param_default
        limit = min(CONF.api_limit_max, limit)
        task_repo = self.gateway.get_task_stub_repo(req.context)
        try:
            tasks = task_repo.list(marker, limit, sort_key, sort_dir, filters)
            if len(tasks) != 0 and len(tasks) == limit:
                result['next_marker'] = tasks[-1].task_id
        except (exception.NotFound, exception.InvalidSortKey, exception.InvalidFilterRangeValue) as e:
            LOG.warning(encodeutils.exception_to_unicode(e))
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.warning(encodeutils.exception_to_unicode(e))
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        result['tasks'] = tasks
        return result

    @debtcollector.removals.remove(message=_DEPRECATION_MESSAGE)
    def get(self, req, task_id):
        _enforce_access_policy(self.policy, req)
        try:
            task_repo = self.gateway.get_task_repo(req.context)
            task = task_repo.get(task_id)
        except exception.NotFound as e:
            msg = _LW('Failed to find task %(task_id)s. Reason: %(reason)s') % {'task_id': task_id, 'reason': encodeutils.exception_to_unicode(e)}
            LOG.warning(msg)
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Forbidden as e:
            msg = _LW('Forbidden to get task %(task_id)s. Reason: %(reason)s') % {'task_id': task_id, 'reason': encodeutils.exception_to_unicode(e)}
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        return task

    @debtcollector.removals.remove(message=_DEPRECATION_MESSAGE)
    def delete(self, req, task_id):
        _enforce_access_policy(self.policy, req)
        msg = _('This operation is currently not permitted on Glance Tasks. They are auto deleted after reaching the time based on their expires_at property.')
        raise webob.exc.HTTPMethodNotAllowed(explanation=msg, headers={'Allow': 'GET'}, body_template='${explanation}')