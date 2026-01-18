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
def _handle_reserved_options(self, argv):
    argv = list(argv)
    for arg, attr in self.reserved_options:
        if arg in argv:
            setattr(self, attr, bool(argv.pop(argv.index(arg))))
    return argv