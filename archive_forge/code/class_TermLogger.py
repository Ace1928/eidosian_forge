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
class TermLogger:
    splash_text = 'celery multi v{version}'
    splash_context = {'version': VERSION_BANNER}
    retcode = 0

    def setup_terminal(self, stdout, stderr, nosplash=False, quiet=False, verbose=False, no_color=False, **kwargs):
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.nosplash = nosplash
        self.quiet = quiet
        self.verbose = verbose
        self.no_color = no_color

    def ok(self, m, newline=True, file=None):
        self.say(m, newline=newline, file=file)
        return EX_OK

    def say(self, m, newline=True, file=None):
        print(m, file=file or self.stdout, end='\n' if newline else '')

    def carp(self, m, newline=True, file=None):
        return self.say(m, newline, file or self.stderr)

    def error(self, msg=None):
        if msg:
            self.carp(msg)
        self.usage()
        return EX_FAILURE

    def info(self, msg, newline=True):
        if self.verbose:
            self.note(msg, newline=newline)

    def note(self, msg, newline=True):
        if not self.quiet:
            self.say(str(msg), newline=newline)

    @splash
    def usage(self):
        self.say(USAGE.format(prog_name=self.prog_name))

    def splash(self):
        if not self.nosplash:
            self.note(self.colored.cyan(self.splash_text.format(**self.splash_context)))

    @cached_property
    def colored(self):
        return term.colored(enabled=not self.no_color)