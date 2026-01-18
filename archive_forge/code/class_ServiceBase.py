import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
class ServiceBase(metaclass=abc.ABCMeta):
    """Base class for all services."""

    @abc.abstractmethod
    def start(self):
        """Start service."""

    @abc.abstractmethod
    def stop(self):
        """Stop service."""

    @abc.abstractmethod
    def wait(self):
        """Wait for service to complete."""

    @abc.abstractmethod
    def reset(self):
        """Reset service.

        Called in case service running in daemon mode receives SIGHUP.
        """