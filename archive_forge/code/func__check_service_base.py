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
def _check_service_base(service):
    if not isinstance(service, ServiceBase):
        raise TypeError(_('Service %(service)s must an instance of %(base)s!') % {'service': service, 'base': ServiceBase})