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
def _cluster_from_argv(self, argv, cmd=None):
    p, nodes = self._nodes_from_argv(argv, cmd=cmd)
    return (p, self.Cluster(list(nodes), cmd=cmd))