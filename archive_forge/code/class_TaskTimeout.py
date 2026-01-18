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
class TaskTimeout(MessagingServerError):
    """Raised if we timed out waiting for a task to complete."""