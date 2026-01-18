import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
class TaskRepoProxy(NotificationRepoProxy, domain_proxy.TaskRepo):

    def get_super_class(self):
        return domain_proxy.TaskRepo

    def get_proxy_class(self):
        return TaskProxy

    def get_payload(self, obj):
        return format_task_notification(obj)

    def add(self, task):
        result = super(TaskRepoProxy, self).add(task)
        self.send_notification('task.create', task)
        return result

    def remove(self, task):
        result = super(TaskRepoProxy, self).remove(task)
        self.send_notification('task.delete', task, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime()})
        return result