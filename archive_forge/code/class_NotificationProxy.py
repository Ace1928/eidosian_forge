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
class NotificationProxy(NotificationBase, metaclass=abc.ABCMeta):

    def __init__(self, repo, context, notifier):
        self.repo = repo
        self.context = context
        self.notifier = notifier
        super_class = self.get_super_class()
        super_class.__init__(self, repo)

    @abc.abstractmethod
    def get_super_class(self):
        pass