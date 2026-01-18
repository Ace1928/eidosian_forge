import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class _CachedArgumentParser(argparse.ArgumentParser):
    """class for caching/collecting command line arguments.

    It also sorts the arguments before initializing the ArgumentParser.
    We need to do this since ArgumentParser by default does not sort
    the argument options and the only way to influence the order of
    arguments in '--help' is to ensure they are added in the sorted
    order.
    """

    def __init__(self, prog=None, usage=None, **kwargs):
        super(_CachedArgumentParser, self).__init__(prog, usage, **kwargs)
        self._args_cache = {}

    def add_parser_argument(self, container, *args, **kwargs):
        values = []
        if container in self._args_cache:
            values = self._args_cache[container]
        values.append({'args': args, 'kwargs': kwargs})
        self._args_cache[container] = values

    def initialize_parser_arguments(self):
        for container, values in self._args_cache.items():
            index = 0
            has_positional = False
            for index, argument in enumerate(values):
                if not argument['args'][0].startswith('-'):
                    has_positional = True
                    break
            size = index if has_positional else len(values)
            values[:size] = sorted(values[:size], key=lambda x: x['args'])
            for argument in values:
                try:
                    container.add_argument(*argument['args'], **argument['kwargs'])
                except argparse.ArgumentError:
                    options = ','.join(argument['args'])
                    raise DuplicateOptError(options)
        self._args_cache = {}

    def parse_args(self, args=None, namespace=None):
        self.initialize_parser_arguments()
        return super(_CachedArgumentParser, self).parse_args(args, namespace)

    def print_help(self, file=None):
        self.initialize_parser_arguments()
        super(_CachedArgumentParser, self).print_help(file)

    def print_usage(self, file=None):
        self.initialize_parser_arguments()
        super(_CachedArgumentParser, self).print_usage(file)