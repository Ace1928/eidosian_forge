import argparse
import os
import sys
from argparse import ArgumentParser, HelpFormatter
from functools import partial
from io import TextIOBase
import django
from django.core import checks
from django.core.exceptions import ImproperlyConfigured
from django.core.management.color import color_style, no_style
from django.db import DEFAULT_DB_ALIAS, connections
class DjangoHelpFormatter(HelpFormatter):
    """
    Customized formatter so that command-specific arguments appear in the
    --help output before arguments common to all commands.
    """
    show_last = {'--version', '--verbosity', '--traceback', '--settings', '--pythonpath', '--no-color', '--force-color', '--skip-checks'}

    def _reordered_actions(self, actions):
        return sorted(actions, key=lambda a: set(a.option_strings) & self.show_last != set())

    def add_usage(self, usage, actions, *args, **kwargs):
        super().add_usage(usage, self._reordered_actions(actions), *args, **kwargs)

    def add_arguments(self, actions):
        super().add_arguments(self._reordered_actions(actions))