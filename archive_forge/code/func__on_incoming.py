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
def _on_incoming(self, incoming):
    """Handles on_incoming event

        :param incoming: incoming request.
        """
    self._work_executor.submit(self._process_incoming, incoming)