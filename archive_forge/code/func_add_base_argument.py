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
def add_base_argument(self, parser, *args, **kwargs):
    """
        Call the parser's add_argument() method, suppressing the help text
        according to BaseCommand.suppressed_base_arguments.
        """
    for arg in args:
        if arg in self.suppressed_base_arguments:
            kwargs['help'] = argparse.SUPPRESS
            break
    parser.add_argument(*args, **kwargs)