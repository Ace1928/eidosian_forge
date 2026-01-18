import abc
import functools
import inspect
import logging
import threading
import traceback
from oslo_config import cfg
from oslo_service import service
from oslo_utils import eventletutils
from oslo_utils import timeutils
from stevedore import driver
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
class ExecutorLoadFailure(MessagingServerError):
    """Raised if an executor can't be loaded."""

    def __init__(self, executor, ex):
        msg = 'Failed to load executor "%s": %s' % (executor, ex)
        super(ExecutorLoadFailure, self).__init__(msg)
        self.executor = executor
        self.ex = ex