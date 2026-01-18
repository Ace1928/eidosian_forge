import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
def _find_sig_argument(self, p, default=signal.SIGTERM):
    args = p.args[len(p.values):]
    for arg in reversed(args):
        if len(arg) == 2 and arg[0] == '-':
            try:
                return int(arg[1])
            except ValueError:
                pass
        if arg[0] == '-':
            try:
                return signals.signum(arg[1:])
            except (AttributeError, TypeError):
                pass
    return default